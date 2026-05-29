# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 quantization kernel implemented in CuTeDSL.

Replicates the core logic of quantize_mxfp8.cuh: given a 2D tensor of BF16/FP16
values, quantize to MXFP8 format (FP8E4M3 data + E8M0 per-block scales).

Matches the C++ kernel's tile dimensions and thread layout:
  CHUNK_DIM_Y = 64, CHUNK_DIM_X = 64, THREADS_PER_CHUNK = 64
  BUFF_DIM_Y  = 32, BUFF_DIM_X  = 64, STAGES = 2
  SCALE_DIM   = 32 (elements per MXFP8 scaling block)

Grid: (ceil(N / 64), ceil(M / 64))
Each block processes a 64x64 chunk in 2 stages of 32x64 tiles loaded into
shared memory.
"""

import os
from re import A
import subprocess
import time

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import Float8TensorStorage
import transformer_engine_torch as tex


from typing import Optional, Type

import cuda.bindings.driver as cuda
import torch
import transformer_engine_torch as tex

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass import Float32, Int64, Int32, Int16, Uint8, Uint32
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.arch import cvt_f32_bf16

import hashlib
import tvm_ffi as _tvm_ffi

from quantize_mxfp8_cutedsl_utils import (
    _ACTIVATIONS,
    FP8E4M3_MAX_NORM_RCP,
    FP8E5M2_MAX_NORM_RCP,
    _bitcast_f32_to_i32,
    _cvt_f32_to_fp8,
    _cvt_f32x2_to_fp8x2,
    _is_packed16,
    _packed16_kit,
    exp2f_rcp,
    fabs_f32,
    float_to_e8m0,
    quantize_colwise_mxfp8,
    quantize_rowwise_mxfp8,
    quantize_rowwise_nvfp4,
    quantize_colwise_nvfp4,
    SCALE_DIM_NVFP4,
)

# MXFP8 settings
MXFP8_BLOCK_SIZE = 32 # Number of elements per MXFP8 scale block. They will share the same E8M0 scale factor
SCALE_DIM = MXFP8_BLOCK_SIZE

# Double-buffering for async copy + compute overlap
BUFFER_NUM = 2

# Vectorised access constants for bank-conflict avoidance (rowwise pass)
PACK_SIZE = 4                              # Elements per vector load
WAVES = SCALE_DIM // PACK_SIZE             # Each thread reads 8 waves with each wave reads 4 packed bf16, so it reads a whole MXFP8 block in total
THREADS_PER_WARP = 32
TOTAL_BANKS_WIDTH = (32 * 4) // 1  # 32 banks × 4 bytes, in bytes (uint8 stride)
THREADS_PER_BANK = TOTAL_BANKS_WIDTH // SCALE_DIM  # 4 threads per bank

# Tiling sizes
NUM_STAGES = 2 # Pipeline depth of the producer/consumer ring buffer for the TMA-G2S input loads (PipelineTmaAsync stage count)
NUM_TILES = 2 # Each CTA process 2 tiles along the Y (row, slowest-changing) dimension
TILE_Y = 32 # Each tile has 32 rows, so each CTA handles 32 * 2 rows in total
TILE_X = 64 # Each tile has 64 columns

# CTA size
THREADS_PER_CHUNK = 64
NUM_WARPS = THREADS_PER_CHUNK // 32

# ---------------------------------------------------------------------------
# Kernel configuration
# ---------------------------------------------------------------------------
class MXFP8QuantizeConfig:
    def __init__(self, dtype, fp8_dtype="e4m3", rowwise=True, colwise=False,
                 with_gemm_swizzled_scales=False, with_amax=False,
                 activation=None, nvfp4_rowwise=False, nvfp4_colwise=False):
        self.DTYPE = dtype
        self.FP8_DTYPE = fp8_dtype
        self.ROWWISE = rowwise
        self.COLWISE = colwise
        # Hybrid extension: optionally also emit NVFP4 in the "other" direction
        # (rowwise MXFP8 + colwise NVFP4, or colwise MXFP8 + rowwise NVFP4).
        # Both NVFP4 passes reuse the same 64-thread CTA and the same input
        # smem tile as the MXFP8 pass — no second DRAM read.
        self.NVFP4_ROWWISE = nvfp4_rowwise
        self.NVFP4_COLWISE = nvfp4_colwise
        self.WITH_GEMM_SWIZZLED_SCALES = with_gemm_swizzled_scales
        self.WITH_AMAX = with_amax
        if activation is not None and activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; expected one of "
                f"{sorted(_ACTIVATIONS)} or None")
        self.ACTIVATION = activation
        self.MAX_NORM_RCP = FP8E4M3_MAX_NORM_RCP if fp8_dtype == "e4m3" else FP8E5M2_MAX_NORM_RCP

# ---------------------------------------------------------------------------
# Unified MXFP8 quantization kernel — shared memory tiled, single-pass
# ---------------------------------------------------------------------------
class MXFP8QuantizeSmemKernel:
    """MXFP8 quantization with shared-memory tiling (rowwise, colwise, or both).

    Matches C++ kernel's BIDIMENSIONAL scaling mode:
      Grid  (ceil(N/64), ceil(M/64))
      Block (64)
      Each block processes a 64x64 chunk in 2 stages of 32x64.

    Per stage, the tile is loaded into shared memory once.  The colwise
    pass reads columns from smem first, then the rowwise pass reads rows.
    When both directions are enabled, global memory is read only once per
    element — matching the C++ single-pass behaviour.

    Thread mappings (per stage):
      Colwise:  thread tidx handles column tidx, 32 rows (stride BUFF_DIM_X).
      Rowwise:  tid_Y = tidx // 2 -> row, tid_X = tidx % 2 -> scale-block.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor, # Input tensor to quantize
        mO_row: Optional[cute.Tensor], mS_row: Optional[cute.Tensor], # Rowwise output and scale tensors
        mO_col: Optional[cute.Tensor], mS_col: Optional[cute.Tensor], # Colwise output and scale tensors
        mAmax: cute.Tensor, # Global amax accumulator, only used in WITH_AMAX path
    ):
        M = mX.shape[0]
        N = mX.shape[1]
        cfg = self.cfg
        max_norm_rcp = cfg.MAX_NORM_RCP
        num_scale_cols = N // SCALE_DIM
        num_scale_rows = M // SCALE_DIM
        
        # Rewrap mS_row / mS_col with the GEMM-swizzled layout when requested.
        # Wrapper passes in a tensor with the compact (M, N/32):(N/32, 1) layout
        # (built from a compact fake-ptr at compile time), and we re-view the
        # underlying buffer here so the per-block scale stores below land at the
        # cuBLAS-swizzled byte offsets.
        # See https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
        # and swizzle_demo.svg for a visual of the byte permutation.
        if cutlass.const_expr(cfg.WITH_GEMM_SWIZZLED_SCALES):
            num_tiles_M = (M + 127) // 128
            num_tiles_SC = (num_scale_cols + 3) // 4   # = ceil(N / 128)
            num_tiles_SR = (num_scale_rows + 3) // 4   # = ceil(M / 128)
            num_tiles_N = (N + 127) // 128
            # row i = i_lo + 32 * (i_hi + 4 * tile_Y);  col j = j_lo + 4 * tile_X.
            # Within one 128×4 tile: byte = i_lo*16 + i_hi*4 + j_lo.
        
            # Tile-major outer dims add (tile_Y * num_tiles_SC + tile_X) * 512.
            # For example, if M=256, N=512, then num_scale_cols = 16, num_scale_rows = 8, and num_tiles_M=2, num_tiles_SC=4, num_tiles_SR=2, num_tiles_N=4
            # The swizzled layout is ((32, 4, 2), (4, 4)):((16, 4, 2048), (1, 512))
            if cutlass.const_expr(cfg.ROWWISE):
                mS_row = cute.make_tensor(
                    mS_row.iterator,
                    cute.make_layout(
                        ((32, 4, num_tiles_M), (4, num_tiles_SC)),
                        stride=((16, 4, num_tiles_SC * 512), (1, 512)),
                    ),
                )
            # Colwise: same swizzle, axes swap roles — col axis gets the 32×4
            # inner decomp, scale-row axis gets the 4-extent dim.
            if cutlass.const_expr(cfg.COLWISE):
                mS_col = cute.make_tensor(
                    mS_col.iterator,
                    cute.make_layout(
                        ((4, num_tiles_SR), (32, 4, num_tiles_N)),
                        stride=((1, 512), (16, 4, num_tiles_SR * 512)),
                    ),
                )
        
        # Divide by the STAGE tile (TILE_Y, TILE_X // SCALE_DIM), not the CTA
        # tile. Each CTA owns NUM_TILES consecutive row-tiles; the kernel walks
        # them by indexing GRID's row dim with `bidy * NUM_TILES + stage` (cute
        # auto-decomposes a flat coord onto GRID's hierarchical row modes).
        #
        # Critically, this is the only divide that cleanly cuts both layouts:
        #   - compact `(M, N/32):(N/32, 1)`  → SCALE_TILE = (32, 2):(N/32, 1)
        #   - swizzled `((32,4,n_M),(4,n_SC)):((16,4,n_SC·512),(1,512))`
        #                                    → SCALE_TILE = (32, 2):(16, 1)
        # The bigger (TILE_Y * NUM_TILES, ...) divide we used before tangles the
        # swizzle's (32, 4) row hierarchy under flatten + sub-divide chain.
        
        # Declare TMA descriptors on the host side.
        # make_tiled_tma_atom returns the UNTILED gmem tensor with basis strides.
        # Tile it inside the kernel with zipped_divide so each coord selects
        # one (TILE_Y, TILE_X) tile.
        smem_tile_layout = cute.make_ordered_layout((TILE_Y, TILE_X), order=(1, 0))
        cta_tiler = (TILE_Y, TILE_X)
        
        # Input: TMA G2S (bf16/fp16 → smem).
        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom, tma_src = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_load, mX, smem_tile_layout, cta_tiler, num_multicast=1,
        )
        
        # Output: TMA S2G (uint8 smem → gmem) for both directions. Creating
        # both atoms unconditionally — if a direction is disabled the kernel
        # simply won't dispatch its copy, and the atom cost is negligible.
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        out_smem_layout = cute.make_ordered_layout((TILE_Y, TILE_X), order=(1, 0))
        tma_atom_out_row = None
        tma_dst_out_row = None
        tma_atom_out_col = None
        tma_dst_out_col = None
        if cutlass.const_expr(cfg.ROWWISE):
            tma_atom_out_row, tma_dst_out_row = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store, mO_row, out_smem_layout, cta_tiler, num_multicast=1,
            )
        if cutlass.const_expr(cfg.COLWISE):
            tma_atom_out_col, tma_dst_out_col = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store, mO_col, out_smem_layout, cta_tiler, num_multicast=1,
            )
        
        # CUDA launches in (0,0), (1,0), (2,0)... order, so we should make N the leading dimension for better access pattern 
        # So consecutive blocks will move along the N dimension first, which is the innermost dimension in memory and we can use cache better
        grid = [
            cute.ceil_div(Int32(N), TILE_X),
            cute.ceil_div(M, TILE_Y * NUM_TILES),
        ]
        block = [THREADS_PER_CHUNK,]
        
        self.kernel(
            mX, mS_row, mS_col, mAmax,
            max_norm_rcp, mX.element_type,
            tma_atom, tma_src,
            tma_atom_out_row, tma_dst_out_row,
            tma_atom_out_col, tma_dst_out_col,
        ).launch(
            grid=grid,
            block=block,
        )

    @cute.kernel
    def kernel(
        self,
        mX,
        mS_row,
        mS_col,
        mAmax, 
        max_norm_rcp,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom, tma_src, # how to use TMA to copy the input
        tma_atom_out_row, tma_dst_out_row, # how to use TMA to copy the rowwise output
        tma_atom_out_col, tma_dst_out_col, # how to use TMA to copy the colwise output
    ):
        cfg = self.cfg

        if cutlass.const_expr(cfg.ROWWISE):
            mS_row = cute.zipped_divide(mS_row, (TILE_Y, TILE_X // SCALE_DIM))
        if cutlass.const_expr(cfg.COLWISE):
            mS_col = cute.zipped_divide(mS_col, (TILE_Y // SCALE_DIM, TILE_X))
        # For M=256, N=512:
        # Non-swizzled: https://kainzhong.github.io/CuTe-Layout-Visualizer/?key=zipped_divide-%28256%2C+16%29%3A%2816%2C+1%29-32%0A2
        # Swizzled: https://kainzhong.github.io/CuTe-Layout-Visualizer/?key=zipped_divide-%28%2832%2C+4%2C+2%29%2C+%284%2C+4%29%29%3A%28%2816%2C+4%2C+2048%29%2C+%281%2C+512%29%29-32%0A2
        # print(f"mS_row after zipped_divide: {mS_row}")

        # FP8 output smem, one 32×64 tile per stage per enabled direction.
        # Allocating a dead sO_col in rowwise-only (or sO_row in colwise-only)
        # bumps per-CTA smem from 12 KB to 16 KB, which drops occupancy and
        # regresses the single-direction path by ~8-10% at 16384^2. Match
        # C++ and only allocate what the active pass actually uses.
        # sAmax holds one f32 per warp for the cross-warp amax reduction —
        # negligible (8 bytes for NUM_WARPS=2) and we always allocate so the
        # struct doesn't fork on a 4th const-expr (cfg.WITH_AMAX) dimension.
        if cutlass.const_expr(cfg.ROWWISE and cfg.COLWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        elif cutlass.const_expr(cfg.ROWWISE and not cfg.COLWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        elif cutlass.const_expr(cfg.ROWWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        else:
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Per-stage shmem tile is 2D (TILE_Y, TILE_X); stages laid out back-to-back.
        # Mode 0 is hierarchical ((TILE_Y, TILE_X),) so it matches the rank/shape
        # of gX_tiled[(None, (ty, tx))] produced by zipped_divide.
        # sX[(None, stage)] selects one (TILE_Y, TILE_X) tile.
        sX = storage.sX.get_tensor(
            cute.make_layout(
                ((TILE_Y, TILE_X), NUM_STAGES),
                stride=((TILE_X, 1), TILE_Y * TILE_X),
            )
        )
        if cutlass.const_expr(cfg.ROWWISE):
            sO_row = storage.sO_row.get_tensor(
                cute.make_layout(
                    ((TILE_Y, TILE_X), NUM_STAGES),
                    stride=((TILE_X, 1), TILE_Y * TILE_X),
                )
            )
        if cutlass.const_expr(cfg.COLWISE):
            sO_col = storage.sO_col.get_tensor(
                cute.make_layout(
                    ((TILE_Y, TILE_X), NUM_STAGES),
                    stride=((TILE_X, 1), TILE_Y * TILE_X),
                )
            )

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptor (one-time; warp-0 only).
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        # Producer: `arrive_and_expect_tx` is wrapped in `elect_one`, so only
        # one lane of warp 0 arrives on the full barrier per stage → arrive_count=1.
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        # Consumer: `consumer_release` arrives only on the `is_signalling_thread`
        # (lane 0 of each warp), so arrive_count = num_warps per stage.
        num_warps = THREADS_PER_CHUNK // 32
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_warps)

        # Bytes transferred per TMA copy: one (TILE_Y, TILE_X) tile of dtype.
        tx_count = TILE_Y * TILE_X * dtype.width // 8

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_storage.data_ptr(),
            num_stages=NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=None,   # single-CTA, no cluster/multicast
        )

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, NUM_STAGES
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, NUM_STAGES
        )

        M = mX.shape[0]
        N = mX.shape[1]

        num_tiles = cutlass.min(
            NUM_TILES,
            cute.ceil_div(M - bidy * TILE_Y * NUM_TILES, TILE_Y),
        )

        # Tile the TMA gmem view: ((TILE_Y, TILE_X), (M/TILE_Y, N/TILE_X)).
        gX_tiled = cute.zipped_divide(tma_src, (TILE_Y, TILE_X))

        # Partition sX/gX for the TMA atom (single-CTA, no cluster/multicast).
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0, # Use the only CTA to do the TMA copy
            cute.make_layout(1), # This cluster only has 1 CTAs
            sX,
            gX_tiled,
        )

        # Same partitioning for S2G outputs: sO_row → mO_row and sO_col → mO_col.
        if cutlass.const_expr(cfg.ROWWISE):
            gO_row_tiled = cute.zipped_divide(tma_dst_out_row, (TILE_Y, TILE_X))
            tXsO_row, tXgO_row = cute.nvgpu.cpasync.tma_partition(
                tma_atom_out_row,
                0,
                cute.make_layout(1),
                sO_row,
                gO_row_tiled,
            )
        if cutlass.const_expr(cfg.COLWISE):
            gO_col_tiled = cute.zipped_divide(tma_dst_out_col, (TILE_Y, TILE_X))
            tXsO_col, tXgO_col = cute.nvgpu.cpasync.tma_partition(
                tma_atom_out_col,
                0,
                cute.make_layout(1),
                sO_col,
                gO_col_tiled,
            )

        # print(f"sX: {sX}\n")
        # print(f"gX_tiled: {gX_tiled}\n")
        # print(f"tXsX: {tXsX}\n")
        # print(f"tXgX: {tXgX}\n")

        # Ensure barrier init is visible to all threads before the pipeline is used.
        cute.arch.sync_threads()

        # ---- Producer: warp 0 issues one TMA copy per tile. ----
        if warp_idx == 0:
            for stage in cutlass.range(num_tiles, unroll=1):
                mainloop_pipeline.producer_acquire(prod_state)
                tile_y = bidy * NUM_TILES + stage
                cute.copy(
                    tma_atom,
                    tXgX[(None, (tile_y, bidx))],
                    tXsX[(None, prod_state.index)],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                )
                mainloop_pipeline.producer_commit(prod_state)
                prod_state.advance()

        # Per-thread amax accumulator across all stages of this CTA. Combined
        # with the per-warp redux + cross-warp shmem reduce + atomic at the
        # bottom to produce a global max(|x|) in mAmax. Initialised to 0
        # since amax is non-negative.
        if cutlass.const_expr(cfg.WITH_AMAX):
            block_amax = Float32(0.0)

        # ---- Consumer: all threads quantize each completed tile. ----
        for stage in cutlass.range(num_tiles, unroll=1):
            mainloop_pipeline.consumer_wait(cons_state)
            sX_tile = sX[(None, stage)]          # (TILE_Y, TILE_X) bf16

            """
            grid = [
                cute.ceil_div(Int32(N), TILE_X),
                cute.ceil_div(M, TILE_Y * NUM_TILES),
            ]
            So to obtain the tile that belongs to this CTA.
            """
            # This is just block's x axis idx
            tile_idx_x = bidx
            # Each CTA has `NUM_TILES` tiles. Each stage we need to obtain the tile for that specific stage. 
            # So the tile index along Y dimension is `bidy * NUM_TILES + stage`
            tile_idx_y = bidy * NUM_TILES + stage
            if cutlass.const_expr(cfg.COLWISE):
                # The first row that belongs to this CTA. Each CTA handles NUM_TILES of (TILE_Y, TILE_X) tiles stacked vertically,
                # and each stage handles one of them.
                sO_col_tile = sO_col[(None, stage)]
                mS_col_stage = cute.flatten(mS_col[(None, (tile_idx_y, tile_idx_x))])

                self._process_colwise(
                    sX_tile, sO_col_tile,
                    mS_col_stage, max_norm_rcp,
                    tile_idx_y * TILE_Y, bidx * TILE_X, M, N,
                )
                amax_c = self._process_colwise(
                    sX_tile, sO_col_tile,
                    mS_col_stage, max_norm_rcp,
                    tile_idx_y * TILE_Y, bidx * TILE_X, M, N,
                )
                if cutlass.const_expr(cfg.WITH_AMAX):
                    block_amax = cute.arch.fmax(block_amax, amax_c)
            if cutlass.const_expr(cfg.ROWWISE):
                sO_row_tile = sO_row[(None, stage)]
                # mS_row is ((SCALE_TILE), (GRID)) where SCALE_TILE = (32, 2).
                # Each CTA owns NUM_TILES consecutive row-tiles of GRID. cute
                # auto-decomposes the flat row coord `bidy * NUM_TILES + stage`
                # onto GRID's hierarchical row modes — which is the
                # (i_hi, tile_Y) tile-major order for swizzled, and the plain
                # row-tile order for compact. Same source, both layouts correct.
                mS_row_stage = cute.flatten(mS_row[(None, (tile_idx_y, tile_idx_x))])
                # print(f"s0_row_tile: {sO_row_tile}\n")
                # print(f"sO_row: {sO_row}\n")
                # print(f"mS_row: {mS_row}\n")
                # print(f"mS_row_stage: {mS_row_stage}\n")
                # print(f"mS_row_stage: {mS_row_stage}\n")
                amax_r = self._process_rowwise(
                    sX_tile, sO_row_tile,
                    mS_row_stage, max_norm_rcp,
                    tile_idx_y * TILE_Y, bidx * TILE_X, M, N,
                )

                if cutlass.const_expr(cfg.WITH_AMAX):
                    block_amax = cute.arch.fmax(block_amax, amax_r)

            # Make all smem stores (sO_row and/or sO_col) visible to the TMA
            # async proxy, then block-sync so warp 0 sees the fences from all
            # warps before issuing the bulk store(s). Matches the C++
            # reference's fence_proxy + __syncthreads pattern.
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            cute.arch.sync_threads()

            if warp_idx == 0:
                tile_y = bidy * NUM_TILES + stage
                if cutlass.const_expr(cfg.ROWWISE):
                    cute.copy(
                        tma_atom_out_row,
                        tXsO_row[(None, stage)],
                        tXgO_row[(None, (tile_y, bidx))],
                    )
                if cutlass.const_expr(cfg.COLWISE):
                    cute.copy(
                        tma_atom_out_col,
                        tXsO_col[(None, stage)],
                        tXgO_col[(None, (tile_y, bidx))],
                    )
                cute.arch.cp_async_bulk_commit_group()

            mainloop_pipeline.consumer_release(cons_state)
            cons_state.advance()

        # Wait for in-flight TMA stores so data is visible to the host
        # before the kernel returns.
        cute.arch.cp_async_bulk_wait_group(0, read=False)

        # ---- amax block reduction + cross-CTA atomic ----------------------
        # 1) intra-warp: redux.sync.fmax.f32 (sm_80+, single instruction).
        # 2) cross-warp: NUM_WARPS shmem floats + sync_threads.
        # 3) cross-CTA: int-atomic-max on the f32 bit pattern. Since amax is
        #    always ≥ 0, IEEE-754 bit ordering on positives matches float
        #    magnitude ordering, so atomic_max on i32 bits gives the right
        #    result. (atomic_max_float32 also exists but its pointer
        #    normalisation is broken as of this CuTeDSL build.)
        if cutlass.const_expr(cfg.WITH_AMAX):
            warp_amax = cute.arch.warp_redux_sync(block_amax, kind="fmax")
            sAmax = storage.sAmax.get_tensor(cute.make_layout(NUM_WARPS))
            lane_idx = tidx % 32
            if lane_idx == 0:
                sAmax[warp_idx] = warp_amax
            cute.arch.sync_threads()
            if tidx == 0:
                cta_amax = Float32(0.0)
                for w in cutlass.range_constexpr(NUM_WARPS):
                    cta_amax = cute.arch.fmax(cta_amax, sAmax[w])
                amax_i32 = cute.make_tensor(
                    cute.recast_ptr(mAmax.iterator, dtype=Int32),
                    cute.make_layout(1),
                )
                cute.arch.atomic_max(
                    amax_i32.iterator, _bitcast_f32_to_i32(cta_amax),
                )

    @cute.jit
    def _process_rowwise(
        self,
        sX_tile,        # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
        sO_row_tile,    # (TILE_Y, TILE_X) uint8 smem view (rowwise FP8 output)
        mS_row_stage,   # rowwise scale tensor (1D swizzled, or 2D linear)
        max_norm_rcp,
        tile_row_start, # Int32 — global row of this stage's row 0
        tile_col_start, # Int32 — global col of this CTA's col 0
        M, N,           # Int32 — full input extents, for OOB masking
    ):
        """Rowwise MXFP8 pass: thread `(tid_Y, tid_X) = (tidx % 32, tidx // 32)`
        owns one 32-element scale block (row `tid_Y`, columns `tid_X*32 .. +32`).

        The bank-group swizzle `((w + bank_group) * PACK_SIZE) % SCALE_DIM`
        staggers each 4-thread group's starting wave, which otherwise would
        collide on smem banks since all lanes in a warp read different rows
        at the same column offset.

        Writes quantized bytes into `sO_row_tile` as u32s (one per wave);
        caller is responsible for the TMA S2G flush.
        """
        cfg = self.cfg
        return quantize_rowwise_mxfp8(
            sX_tile,
            sO_row_tile,
            mS_row_stage,
            max_norm_rcp,
            tile_row_start,
            tile_col_start,
            M,
            N,
            ACTIVATION=cfg.ACTIVATION,
            DTYPE=cfg.DTYPE,
            ROWWISE=cfg.ROWWISE,
            COLWISE=cfg.COLWISE,
            FP8_DTYPE=cfg.FP8_DTYPE,
            TILE_Y=TILE_Y,
            SCALE_DIM=SCALE_DIM,
            WAVES=WAVES,
            THREADS_PER_WARP=THREADS_PER_WARP,
            THREADS_PER_BANK=THREADS_PER_BANK,
            PACK_SIZE=PACK_SIZE
        )

    @cute.jit
    def _process_colwise(
        self,
        sX_tile,        # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
        sO_col_tile,    # (TILE_Y, TILE_X) uint8 smem view (colwise FP8 output)
        mS_col_stage,   # colwise scale tensor (1D swizzled, or 2D linear)
        max_norm_rcp,
        tile_row_start, # Int32 — global row of this stage's row 0
        tile_col_start, # Int32 — global col of this CTA's col 0
        M, N,           # Int32 — full input extents, for OOB masking
    ):
        """Colwise MXFP8 pass: thread `tidx` owns column `tidx` of the (32, 64)
        smem tile — 32 elements down. Writes quantized bytes into `sO_col_tile`
        so the caller can flush with a TMA S2G — matches C++'s
        `out_colwise_data_sh` + `cp.async.bulk.tensor.2d.shared_to_global`.
        """
        cfg = self.cfg
        return quantize_colwise_mxfp8(
            sX_tile,
            sO_col_tile,
            mS_col_stage,
            max_norm_rcp,
            tile_row_start,
            tile_col_start,
            M,
            N,
            ACTIVATION=cfg.ACTIVATION,
            DTYPE=cfg.DTYPE,
            FP8_DTYPE=cfg.FP8_DTYPE,
            SWIZZLE=cfg.WITH_GEMM_SWIZZLED_SCALES,
            TILE_X=TILE_X,
            TILE_Y=TILE_Y,
            SCALE_DIM=SCALE_DIM,
        )



class HybridQuantizeSmemKernel(MXFP8QuantizeSmemKernel):
    """MXFP8 in one direction + NVFP4 in the other, from a single shared-memory
    read of the input tile. Inherits __init__ / _process_rowwise / _process_colwise
    from MXFP8QuantizeSmemKernel; overrides __call__ (11 params) and kernel to add
    the NVFP4 passes. Kept separate so the MXFP8-only tvm-ffi entry stays 6-param
    (its --enable-tvm-ffi wrapper cost scales with declared param count)."""

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor, # Input tensor to quantize
        mO_row: Optional[cute.Tensor], mS_row: Optional[cute.Tensor], # Rowwise MXFP8 output and scale tensors
        mO_col: Optional[cute.Tensor], mS_col: Optional[cute.Tensor], # Colwise MXFP8 output and scale tensors
        mAmax: cute.Tensor, # Global amax accumulator, only used in WITH_AMAX path
        # --- Hybrid NVFP4 outputs (all None unless the matching cfg flag is set) ---
        # Data tiles are written to smem by the NVFP4 utils and TMA-flushed here;
        # scale tensors (E4M3 bytes) are written straight to gmem by the utils.
        #   rowwise NVFP4: mO_nvfp4_row (M, N//2),  mS_nvfp4_row (M, N//16)
        #   colwise NVFP4: mO_nvfp4_col (N, M//2),  mS_nvfp4_col (N, M//16) (transposed)
        mO_nvfp4_row: Optional[cute.Tensor] = None, mS_nvfp4_row: Optional[cute.Tensor] = None,
        mO_nvfp4_col: Optional[cute.Tensor] = None, mS_nvfp4_col: Optional[cute.Tensor] = None,
        # Global NVFP4 encode scale S_enc = 448*6/global_amax, host-precomputed
        # and passed as a runtime Float32 scalar (None when no NVFP4 direction).
        s_enc=None,
    ):
        M = mX.shape[0]
        N = mX.shape[1]
        cfg = self.cfg
        max_norm_rcp = cfg.MAX_NORM_RCP
        num_scale_cols = N // SCALE_DIM
        num_scale_rows = M // SCALE_DIM
        
        # Rewrap mS_row / mS_col with the GEMM-swizzled layout when requested.
        # Wrapper passes in a tensor with the compact (M, N/32):(N/32, 1) layout
        # (built from a compact fake-ptr at compile time), and we re-view the
        # underlying buffer here so the per-block scale stores below land at the
        # cuBLAS-swizzled byte offsets.
        # See https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
        # and swizzle_demo.svg for a visual of the byte permutation.
        if cutlass.const_expr(cfg.WITH_GEMM_SWIZZLED_SCALES):
            num_tiles_M = (M + 127) // 128
            num_tiles_SC = (num_scale_cols + 3) // 4   # = ceil(N / 128)
            num_tiles_SR = (num_scale_rows + 3) // 4   # = ceil(M / 128)
            num_tiles_N = (N + 127) // 128
            # row i = i_lo + 32 * (i_hi + 4 * tile_Y);  col j = j_lo + 4 * tile_X.
            # Within one 128×4 tile: byte = i_lo*16 + i_hi*4 + j_lo.
        
            # Tile-major outer dims add (tile_Y * num_tiles_SC + tile_X) * 512.
            # For example, if M=256, N=512, then num_scale_cols = 16, num_scale_rows = 8, and num_tiles_M=2, num_tiles_SC=4, num_tiles_SR=2, num_tiles_N=4
            # The swizzled layout is ((32, 4, 2), (4, 4)):((16, 4, 2048), (1, 512))
            if cutlass.const_expr(cfg.ROWWISE):
                mS_row = cute.make_tensor(
                    mS_row.iterator,
                    cute.make_layout(
                        ((32, 4, num_tiles_M), (4, num_tiles_SC)),
                        stride=((16, 4, num_tiles_SC * 512), (1, 512)),
                    ),
                )
            # Colwise: same swizzle, axes swap roles — col axis gets the 32×4
            # inner decomp, scale-row axis gets the 4-extent dim.
            if cutlass.const_expr(cfg.COLWISE):
                mS_col = cute.make_tensor(
                    mS_col.iterator,
                    cute.make_layout(
                        ((4, num_tiles_SR), (32, 4, num_tiles_N)),
                        stride=((1, 512), (16, 4, num_tiles_SR * 512)),
                    ),
                )
        
        # Divide by the STAGE tile (TILE_Y, TILE_X // SCALE_DIM), not the CTA
        # tile. Each CTA owns NUM_TILES consecutive row-tiles; the kernel walks
        # them by indexing GRID's row dim with `bidy * NUM_TILES + stage` (cute
        # auto-decomposes a flat coord onto GRID's hierarchical row modes).
        #
        # Critically, this is the only divide that cleanly cuts both layouts:
        #   - compact `(M, N/32):(N/32, 1)`  → SCALE_TILE = (32, 2):(N/32, 1)
        #   - swizzled `((32,4,n_M),(4,n_SC)):((16,4,n_SC·512),(1,512))`
        #                                    → SCALE_TILE = (32, 2):(16, 1)
        # The bigger (TILE_Y * NUM_TILES, ...) divide we used before tangles the
        # swizzle's (32, 4) row hierarchy under flatten + sub-divide chain.
        
        # Declare TMA descriptors on the host side.
        # make_tiled_tma_atom returns the UNTILED gmem tensor with basis strides.
        # Tile it inside the kernel with zipped_divide so each coord selects
        # one (TILE_Y, TILE_X) tile.
        smem_tile_layout = cute.make_ordered_layout((TILE_Y, TILE_X), order=(1, 0))
        cta_tiler = (TILE_Y, TILE_X)
        
        # Input: TMA G2S (bf16/fp16 → smem).
        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom, tma_src = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_load, mX, smem_tile_layout, cta_tiler, num_multicast=1,
        )
        
        # Output: TMA S2G (uint8 smem → gmem) for both directions. Creating
        # both atoms unconditionally — if a direction is disabled the kernel
        # simply won't dispatch its copy, and the atom cost is negligible.
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        out_smem_layout = cute.make_ordered_layout((TILE_Y, TILE_X), order=(1, 0))
        tma_atom_out_row = None
        tma_dst_out_row = None
        tma_atom_out_col = None
        tma_dst_out_col = None
        if cutlass.const_expr(cfg.ROWWISE):
            tma_atom_out_row, tma_dst_out_row = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store, mO_row, out_smem_layout, cta_tiler, num_multicast=1,
            )
        if cutlass.const_expr(cfg.COLWISE):
            tma_atom_out_col, tma_dst_out_col = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store, mO_col, out_smem_layout, cta_tiler, num_multicast=1,
            )

        # NVFP4 S2G store atoms. fp4 packs 2 elements per byte, so the output
        # tiles are half-width in the packed dimension:
        #   rowwise: (TILE_Y, TILE_X // 2)   — pairs packed along X
        #   colwise: (TILE_X, TILE_Y // 2)   — TRANSPOSED, pairs packed along
        #            the (input-)row axis to match TE's NVFP4 columnwise layout.
        tma_atom_nvfp4_row = None
        tma_dst_nvfp4_row = None
        tma_atom_nvfp4_col = None
        tma_dst_nvfp4_col = None
        if cutlass.const_expr(cfg.NVFP4_ROWWISE):
            nvfp4_row_tiler = (TILE_Y, TILE_X // 2)
            nvfp4_row_smem_layout = cute.make_ordered_layout(nvfp4_row_tiler, order=(1, 0))
            tma_atom_nvfp4_row, tma_dst_nvfp4_row = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store, mO_nvfp4_row, nvfp4_row_smem_layout, nvfp4_row_tiler, num_multicast=1,
            )
        if cutlass.const_expr(cfg.NVFP4_COLWISE):
            nvfp4_col_tiler = (TILE_X, TILE_Y // 2)
            nvfp4_col_smem_layout = cute.make_ordered_layout(nvfp4_col_tiler, order=(1, 0))
            tma_atom_nvfp4_col, tma_dst_nvfp4_col = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store, mO_nvfp4_col, nvfp4_col_smem_layout, nvfp4_col_tiler, num_multicast=1,
            )

        # CUDA launches in (0,0), (1,0), (2,0)... order, so we should make N the leading dimension for better access pattern 
        # So consecutive blocks will move along the N dimension first, which is the innermost dimension in memory and we can use cache better
        grid = [
            cute.ceil_div(Int32(N), TILE_X),
            cute.ceil_div(M, TILE_Y * NUM_TILES),
        ]
        block = [THREADS_PER_CHUNK,]
        
        self.kernel(
            mX, mS_row, mS_col, mAmax,
            max_norm_rcp, mX.element_type,
            tma_atom, tma_src,
            tma_atom_out_row, tma_dst_out_row,
            tma_atom_out_col, tma_dst_out_col,
            mS_nvfp4_row, mS_nvfp4_col,
            tma_atom_nvfp4_row, tma_dst_nvfp4_row,
            tma_atom_nvfp4_col, tma_dst_nvfp4_col,
            s_enc,
        ).launch(
            grid=grid,
            block=block,
        )


    @cute.kernel
    def kernel(
        self,
        mX,
        mS_row,
        mS_col,
        mAmax, 
        max_norm_rcp,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom, tma_src, # how to use TMA to copy the input
        tma_atom_out_row, tma_dst_out_row, # how to use TMA to copy the rowwise output
        tma_atom_out_col, tma_dst_out_col, # how to use TMA to copy the colwise output
        mS_nvfp4_row, mS_nvfp4_col, # NVFP4 scale tensors (E4M3 bytes, written direct to gmem)
        tma_atom_nvfp4_row, tma_dst_nvfp4_row, # NVFP4 rowwise output S2G
        tma_atom_nvfp4_col, tma_dst_nvfp4_col, # NVFP4 colwise output S2G
        s_enc, # NVFP4 global encode scale, runtime Float32 (None if no NVFP4)
    ):
        cfg = self.cfg

        if cutlass.const_expr(cfg.ROWWISE):
            mS_row = cute.zipped_divide(mS_row, (TILE_Y, TILE_X // SCALE_DIM))
        if cutlass.const_expr(cfg.COLWISE):
            mS_col = cute.zipped_divide(mS_col, (TILE_Y // SCALE_DIM, TILE_X))
        # NVFP4 scale tensors (E4M3 bytes). Rowwise scale is (M, N//16) → per-stage
        # tile (TILE_Y, TILE_X//16); colwise scale is the transposed (N, M//16) →
        # per-stage tile (TILE_X, TILE_Y//16).
        if cutlass.const_expr(cfg.NVFP4_ROWWISE):
            mS_nvfp4_row = cute.zipped_divide(mS_nvfp4_row, (TILE_Y, TILE_X // SCALE_DIM_NVFP4))
        if cutlass.const_expr(cfg.NVFP4_COLWISE):
            mS_nvfp4_col = cute.zipped_divide(mS_nvfp4_col, (TILE_X, TILE_Y // SCALE_DIM_NVFP4))
        # For M=256, N=512:
        # Non-swizzled: https://kainzhong.github.io/CuTe-Layout-Visualizer/?key=zipped_divide-%28256%2C+16%29%3A%2816%2C+1%29-32%0A2
        # Swizzled: https://kainzhong.github.io/CuTe-Layout-Visualizer/?key=zipped_divide-%28%2832%2C+4%2C+2%29%2C+%284%2C+4%29%29%3A%28%2816%2C+4%2C+2048%29%2C+%281%2C+512%29%29-32%0A2
        # print(f"mS_row after zipped_divide: {mS_row}")

        # FP8 output smem, one 32×64 tile per stage per enabled direction.
        # Allocating a dead sO_col in rowwise-only (or sO_row in colwise-only)
        # bumps per-CTA smem from 12 KB to 16 KB, which drops occupancy and
        # regresses the single-direction path by ~8-10% at 16384^2. Match
        # C++ and only allocate what the active pass actually uses.
        # sAmax holds one f32 per warp for the cross-warp amax reduction —
        # negligible (8 bytes for NUM_WARPS=2) and we always allocate so the
        # struct doesn't fork on a 4th const-expr (cfg.WITH_AMAX) dimension.
        if cutlass.const_expr(cfg.ROWWISE and cfg.COLWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        elif cutlass.const_expr(cfg.ROWWISE and not cfg.COLWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        elif cutlass.const_expr(cfg.ROWWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        else:
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Per-stage shmem tile is 2D (TILE_Y, TILE_X); stages laid out back-to-back.
        # Mode 0 is hierarchical ((TILE_Y, TILE_X),) so it matches the rank/shape
        # of gX_tiled[(None, (ty, tx))] produced by zipped_divide.
        # sX[(None, stage)] selects one (TILE_Y, TILE_X) tile.
        sX = storage.sX.get_tensor(
            cute.make_layout(
                ((TILE_Y, TILE_X), NUM_STAGES),
                stride=((TILE_X, 1), TILE_Y * TILE_X),
            )
        )
        if cutlass.const_expr(cfg.ROWWISE):
            sO_row = storage.sO_row.get_tensor(
                cute.make_layout(
                    ((TILE_Y, TILE_X), NUM_STAGES),
                    stride=((TILE_X, 1), TILE_Y * TILE_X),
                )
            )
        if cutlass.const_expr(cfg.COLWISE):
            sO_col = storage.sO_col.get_tensor(
                cute.make_layout(
                    ((TILE_Y, TILE_X), NUM_STAGES),
                    stride=((TILE_X, 1), TILE_Y * TILE_X),
                )
            )

        # NVFP4 output smem tiles (uint8, 2 fp4 per byte). Allocated AFTER the
        # MXFP8 SharedStorage struct via the same allocator so the kernel only
        # reserves them when a hybrid cfg is active — the MXFP8-only forks above
        # are byte-for-byte unchanged.
        if cutlass.const_expr(cfg.NVFP4_ROWWISE):
            sO_nvfp4_row = smem.allocate_tensor(
                Uint8,
                cute.make_layout(
                    ((TILE_Y, TILE_X // 2), NUM_STAGES),
                    stride=((TILE_X // 2, 1), TILE_Y * (TILE_X // 2)),
                ),
                byte_alignment=128,
            )
        if cutlass.const_expr(cfg.NVFP4_COLWISE):
            sO_nvfp4_col = smem.allocate_tensor(
                Uint8,
                cute.make_layout(
                    ((TILE_X, TILE_Y // 2), NUM_STAGES),
                    stride=((TILE_Y // 2, 1), TILE_X * (TILE_Y // 2)),
                ),
                byte_alignment=128,
            )

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptor (one-time; warp-0 only).
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        # Producer: `arrive_and_expect_tx` is wrapped in `elect_one`, so only
        # one lane of warp 0 arrives on the full barrier per stage → arrive_count=1.
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        # Consumer: `consumer_release` arrives only on the `is_signalling_thread`
        # (lane 0 of each warp), so arrive_count = num_warps per stage.
        num_warps = THREADS_PER_CHUNK // 32
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_warps)

        # Bytes transferred per TMA copy: one (TILE_Y, TILE_X) tile of dtype.
        tx_count = TILE_Y * TILE_X * dtype.width // 8

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_storage.data_ptr(),
            num_stages=NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=None,   # single-CTA, no cluster/multicast
        )

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, NUM_STAGES
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, NUM_STAGES
        )

        M = mX.shape[0]
        N = mX.shape[1]

        num_tiles = cutlass.min(
            NUM_TILES,
            cute.ceil_div(M - bidy * TILE_Y * NUM_TILES, TILE_Y),
        )

        # Tile the TMA gmem view: ((TILE_Y, TILE_X), (M/TILE_Y, N/TILE_X)).
        gX_tiled = cute.zipped_divide(tma_src, (TILE_Y, TILE_X))

        # Partition sX/gX for the TMA atom (single-CTA, no cluster/multicast).
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0, # Use the only CTA to do the TMA copy
            cute.make_layout(1), # This cluster only has 1 CTAs
            sX,
            gX_tiled,
        )

        # Same partitioning for S2G outputs: sO_row → mO_row and sO_col → mO_col.
        if cutlass.const_expr(cfg.ROWWISE):
            gO_row_tiled = cute.zipped_divide(tma_dst_out_row, (TILE_Y, TILE_X))
            tXsO_row, tXgO_row = cute.nvgpu.cpasync.tma_partition(
                tma_atom_out_row,
                0,
                cute.make_layout(1),
                sO_row,
                gO_row_tiled,
            )
        if cutlass.const_expr(cfg.COLWISE):
            gO_col_tiled = cute.zipped_divide(tma_dst_out_col, (TILE_Y, TILE_X))
            tXsO_col, tXgO_col = cute.nvgpu.cpasync.tma_partition(
                tma_atom_out_col,
                0,
                cute.make_layout(1),
                sO_col,
                gO_col_tiled,
            )
        # NVFP4 output S2G partitioning (mirrors the MXFP8 outputs, half-width
        # tiles). gO outer dims: rowwise (M/TILE_Y, N/TILE_X), colwise
        # (N/TILE_X, M/TILE_Y) — same (tile_y, tile_x) indexing as MXFP8.
        if cutlass.const_expr(cfg.NVFP4_ROWWISE):
            gO_nvfp4_row_tiled = cute.zipped_divide(tma_dst_nvfp4_row, (TILE_Y, TILE_X // 2))
            tXsO_nvfp4_row, tXgO_nvfp4_row = cute.nvgpu.cpasync.tma_partition(
                tma_atom_nvfp4_row, 0, cute.make_layout(1), sO_nvfp4_row, gO_nvfp4_row_tiled,
            )
        if cutlass.const_expr(cfg.NVFP4_COLWISE):
            gO_nvfp4_col_tiled = cute.zipped_divide(tma_dst_nvfp4_col, (TILE_X, TILE_Y // 2))
            tXsO_nvfp4_col, tXgO_nvfp4_col = cute.nvgpu.cpasync.tma_partition(
                tma_atom_nvfp4_col, 0, cute.make_layout(1), sO_nvfp4_col, gO_nvfp4_col_tiled,
            )
        # Bake the NVFP4 global encode scale into an f32 register once.
        if cutlass.const_expr(cfg.NVFP4_ROWWISE or cfg.NVFP4_COLWISE):
            S_enc_f32 = Float32(s_enc)

        # print(f"sX: {sX}\n")
        # print(f"gX_tiled: {gX_tiled}\n")
        # print(f"tXsX: {tXsX}\n")
        # print(f"tXgX: {tXgX}\n")

        # Ensure barrier init is visible to all threads before the pipeline is used.
        cute.arch.sync_threads()

        # ---- Producer: warp 0 issues one TMA copy per tile. ----
        if warp_idx == 0:
            for stage in cutlass.range(num_tiles, unroll=1):
                mainloop_pipeline.producer_acquire(prod_state)
                tile_y = bidy * NUM_TILES + stage
                cute.copy(
                    tma_atom,
                    tXgX[(None, (tile_y, bidx))],
                    tXsX[(None, prod_state.index)],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                )
                mainloop_pipeline.producer_commit(prod_state)
                prod_state.advance()

        # Per-thread amax accumulator across all stages of this CTA. Combined
        # with the per-warp redux + cross-warp shmem reduce + atomic at the
        # bottom to produce a global max(|x|) in mAmax. Initialised to 0
        # since amax is non-negative.
        if cutlass.const_expr(cfg.WITH_AMAX):
            block_amax = Float32(0.0)

        # ---- Consumer: all threads quantize each completed tile. ----
        for stage in cutlass.range(num_tiles, unroll=1):
            mainloop_pipeline.consumer_wait(cons_state)
            sX_tile = sX[(None, stage)]          # (TILE_Y, TILE_X) bf16

            """
            grid = [
                cute.ceil_div(Int32(N), TILE_X),
                cute.ceil_div(M, TILE_Y * NUM_TILES),
            ]
            So to obtain the tile that belongs to this CTA.
            """
            # This is just block's x axis idx
            tile_idx_x = bidx
            # Each CTA has `NUM_TILES` tiles. Each stage we need to obtain the tile for that specific stage. 
            # So the tile index along Y dimension is `bidy * NUM_TILES + stage`
            tile_idx_y = bidy * NUM_TILES + stage
            if cutlass.const_expr(cfg.COLWISE):
                # The first row that belongs to this CTA. Each CTA handles NUM_TILES of (TILE_Y, TILE_X) tiles stacked vertically,
                # and each stage handles one of them.
                sO_col_tile = sO_col[(None, stage)]
                mS_col_stage = cute.flatten(mS_col[(None, (tile_idx_y, tile_idx_x))])

                self._process_colwise(
                    sX_tile, sO_col_tile,
                    mS_col_stage, max_norm_rcp,
                    tile_idx_y * TILE_Y, bidx * TILE_X, M, N,
                )
                amax_c = self._process_colwise(
                    sX_tile, sO_col_tile,
                    mS_col_stage, max_norm_rcp,
                    tile_idx_y * TILE_Y, bidx * TILE_X, M, N,
                )
                if cutlass.const_expr(cfg.WITH_AMAX):
                    block_amax = cute.arch.fmax(block_amax, amax_c)
            if cutlass.const_expr(cfg.ROWWISE):
                sO_row_tile = sO_row[(None, stage)]
                # mS_row is ((SCALE_TILE), (GRID)) where SCALE_TILE = (32, 2).
                # Each CTA owns NUM_TILES consecutive row-tiles of GRID. cute
                # auto-decomposes the flat row coord `bidy * NUM_TILES + stage`
                # onto GRID's hierarchical row modes — which is the
                # (i_hi, tile_Y) tile-major order for swizzled, and the plain
                # row-tile order for compact. Same source, both layouts correct.
                mS_row_stage = cute.flatten(mS_row[(None, (tile_idx_y, tile_idx_x))])
                # print(f"s0_row_tile: {sO_row_tile}\n")
                # print(f"sO_row: {sO_row}\n")
                # print(f"mS_row: {mS_row}\n")
                # print(f"mS_row_stage: {mS_row_stage}\n")
                # print(f"mS_row_stage: {mS_row_stage}\n")
                amax_r = self._process_rowwise(
                    sX_tile, sO_row_tile,
                    mS_row_stage, max_norm_rcp,
                    tile_idx_y * TILE_Y, bidx * TILE_X, M, N,
                )

                if cutlass.const_expr(cfg.WITH_AMAX):
                    block_amax = cute.arch.fmax(block_amax, amax_r)

            # ---- NVFP4 passes on the SAME smem input tile (no re-read) ----
            # These quantize the identical sX_tile in the "other" direction and
            # write fp4 bytes into their own smem tiles + E4M3 scale bytes
            # straight to gmem. Flushed by the warp-0 TMA block below.
            if cutlass.const_expr(cfg.NVFP4_ROWWISE):
                sO_nvfp4_row_tile = sO_nvfp4_row[(None, stage)]
                mS_nvfp4_row_stage = cute.flatten(
                    mS_nvfp4_row[(None, (tile_idx_y, tile_idx_x))])
                quantize_rowwise_nvfp4(
                    sX_tile, sO_nvfp4_row_tile, mS_nvfp4_row_stage, S_enc_f32,
                    tile_idx_y * TILE_Y, bidx * TILE_X, M, N,
                    DTYPE=cfg.DTYPE, TILE_Y=TILE_Y, TILE_X=TILE_X,
                    SCALE_DIM=SCALE_DIM_NVFP4,
                )
            if cutlass.const_expr(cfg.NVFP4_COLWISE):
                sO_nvfp4_col_tile = sO_nvfp4_col[(None, stage)]
                # Colwise scale tensor is transposed (N, M//16) → outer index
                # (tile_idx_x, tile_idx_y).
                mS_nvfp4_col_stage = cute.flatten(
                    mS_nvfp4_col[(None, (tile_idx_x, tile_idx_y))])
                quantize_colwise_nvfp4(
                    sX_tile, sO_nvfp4_col_tile, mS_nvfp4_col_stage, S_enc_f32,
                    tile_idx_y * TILE_Y, bidx * TILE_X, M, N,
                    DTYPE=cfg.DTYPE, TILE_X=TILE_X, TILE_Y=TILE_Y,
                    SCALE_DIM=SCALE_DIM_NVFP4,
                )

            # Make all smem stores (sO_row and/or sO_col, NVFP4 tiles) visible to
            # the TMA async proxy, then block-sync so warp 0 sees the fences from
            # all warps before issuing the bulk store(s). Matches the C++
            # reference's fence_proxy + __syncthreads pattern.
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            cute.arch.sync_threads()

            if warp_idx == 0:
                tile_y = bidy * NUM_TILES + stage
                if cutlass.const_expr(cfg.ROWWISE):
                    cute.copy(
                        tma_atom_out_row,
                        tXsO_row[(None, stage)],
                        tXgO_row[(None, (tile_y, bidx))],
                    )
                if cutlass.const_expr(cfg.COLWISE):
                    cute.copy(
                        tma_atom_out_col,
                        tXsO_col[(None, stage)],
                        tXgO_col[(None, (tile_y, bidx))],
                    )
                # NVFP4 stores. Rowwise out (M, N//2) indexed (tile_y, bidx);
                # colwise out (N, M//2) indexed (bidx, tile_y) (transposed).
                if cutlass.const_expr(cfg.NVFP4_ROWWISE):
                    cute.copy(
                        tma_atom_nvfp4_row,
                        tXsO_nvfp4_row[(None, stage)],
                        tXgO_nvfp4_row[(None, (tile_y, bidx))],
                    )
                if cutlass.const_expr(cfg.NVFP4_COLWISE):
                    cute.copy(
                        tma_atom_nvfp4_col,
                        tXsO_nvfp4_col[(None, stage)],
                        tXgO_nvfp4_col[(None, (bidx, tile_y))],
                    )
                cute.arch.cp_async_bulk_commit_group()

            mainloop_pipeline.consumer_release(cons_state)
            cons_state.advance()

        # Wait for in-flight TMA stores so data is visible to the host
        # before the kernel returns.
        cute.arch.cp_async_bulk_wait_group(0, read=False)

        # ---- amax block reduction + cross-CTA atomic ----------------------
        # 1) intra-warp: redux.sync.fmax.f32 (sm_80+, single instruction).
        # 2) cross-warp: NUM_WARPS shmem floats + sync_threads.
        # 3) cross-CTA: int-atomic-max on the f32 bit pattern. Since amax is
        #    always ≥ 0, IEEE-754 bit ordering on positives matches float
        #    magnitude ordering, so atomic_max on i32 bits gives the right
        #    result. (atomic_max_float32 also exists but its pointer
        #    normalisation is broken as of this CuTeDSL build.)
        if cutlass.const_expr(cfg.WITH_AMAX):
            warp_amax = cute.arch.warp_redux_sync(block_amax, kind="fmax")
            sAmax = storage.sAmax.get_tensor(cute.make_layout(NUM_WARPS))
            lane_idx = tidx % 32
            if lane_idx == 0:
                sAmax[warp_idx] = warp_amax
            cute.arch.sync_threads()
            if tidx == 0:
                cta_amax = Float32(0.0)
                for w in cutlass.range_constexpr(NUM_WARPS):
                    cta_amax = cute.arch.fmax(cta_amax, sAmax[w])
                amax_i32 = cute.make_tensor(
                    cute.recast_ptr(mAmax.iterator, dtype=Int32),
                    cute.make_layout(1),
                )
                cute.arch.atomic_max(
                    amax_i32.iterator, _bitcast_f32_to_i32(cta_amax),
                )


def _cfg_to_fn_name(cfg, M, N) -> str:
    """Deterministic registry key from (cfg, shape)."""
    key = (cfg.DTYPE.__name__, cfg.FP8_DTYPE,
           int(cfg.ROWWISE), int(cfg.COLWISE),
           int(cfg.WITH_GEMM_SWIZZLED_SCALES), int(cfg.WITH_AMAX),
           cfg.ACTIVATION or "none",
           M, N)
    h = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
    return f"mxfp8_{h}"

_compile_cache_tvm_ffi: dict = {}

def _get_compiled_kernel(cfg, M, N):
    """Compile the kernel for THIS (cfg, M, N) with LITERAL shapes — every
    dimension is a constexpr int, so the AOT wrapper's per-arg type collapses
    to `{ void* data; }` (no shape array, no shape check at call time).

    Tradeoff vs sym_int: one compile per (cfg, M, N) instead of one per cfg.
    Memory cost is small; the per-call saving is ~7-8 us. Cache key already
    includes (M, N) so we never recompile."""
    cache = _compile_cache_tvm_ffi
    fn_name = _cfg_to_fn_name(cfg, M, N)
    if fn_name in cache:
        return fn_name

    kernel_obj = MXFP8QuantizeSmemKernel(cfg)

    # TE allocates scale tensors at this padded shape regardless of swizzle
    # (see MXFP8Quantizer::get_scale_shape in transformer_engine/pytorch/csrc):
    #   rowwise:    (roundup(M, 128),    roundup(N // 32, 4))
    #   columnwise: (roundup(M // 32, 4), roundup(N, 128))
    SCALE_R = (((M + 127) // 128) * 128, ((N + 127) // 128) * 4)
    SCALE_C = (((M + 127) // 128) * 4,   ((N + 127) // 128) * 128)
    WS_M = (M + TILE_Y * NUM_TILES - 1) // (TILE_Y * NUM_TILES)

    # stride_order=(1, 0): row-major, dim 1 stride 1. 1D: (0,).
    kw_rm16_2d = dict(stride_order=(1, 0),
                      memspace=cute.AddressSpace.gmem, assumed_align=16)
    kw_rm4_2d  = dict(stride_order=(1, 0),
                      memspace=cute.AddressSpace.gmem, assumed_align=4)
    kw_rm4_1d  = dict(stride_order=(0,),
                      memspace=cute.AddressSpace.gmem, assumed_align=4)
    def fake(dtype, shape, kw):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, **kw)

    in_fake        = fake(cfg.DTYPE,  (M, N),    kw_rm16_2d)
    out_row_fake   = fake(cute.Uint8, (M, N),    kw_rm16_2d) if cfg.ROWWISE   else None
    scale_row_fake = fake(cute.Uint8, SCALE_R,   kw_rm16_2d) if cfg.ROWWISE   else None
    out_col_fake   = fake(cute.Uint8, (M, N),    kw_rm16_2d) if cfg.COLWISE   else None
    scale_col_fake = fake(cute.Uint8, SCALE_C,   kw_rm16_2d) if cfg.COLWISE   else None
    amax_fake = fake(Float32, (1,), kw_rm4_1d) if cfg.WITH_AMAX else None

    compiled = cute.compile(
        kernel_obj,
        in_fake,                            # mX
        out_row_fake,   scale_row_fake,     # mO_row, mS_row
        out_col_fake,   scale_col_fake,     # mO_col, mS_col
        amax_fake,                          # mAmax
        options="--enable-tvm-ffi",
    )
    _tvm_ffi.register_global_func(fn_name, compiled, override=True)
    cache[fn_name] = compiled
    return fn_name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_torch_to_cutlass_dtype = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


dummy_float32 = torch.zeros(1, dtype=torch.float32, device="cuda")
dummy_uint8 = torch.zeros(1, dtype=torch.uint8, device="cuda")
dummy_cute_uint8 = cute.runtime.make_fake_compact_tensor(cute.Uint8, (1,), memspace=cute.AddressSpace.gmem, assumed_align=4)
dummy_cute_float32 = cute.runtime.make_fake_compact_tensor(Float32, (1,), memspace=cute.AddressSpace.gmem, assumed_align=4)


_TVM_FFI_DUMMY_CACHE: dict = {}
_FP8_DTYPES = {
    "e4m3": tex.DType.kFloat8E4M3,
    "e5m2": tex.DType.kFloat8E5M2,
}

def get_quantize_mxfp8_cutedsl_func(
    x: torch.Tensor,
    fp8_dtype: str = "e4m3",
    rowwise: bool = True,
    colwise: bool = False,
    with_gemm_swizzled_scales: bool = False,
    with_amax: bool = False,
    activation: str = None,
) -> str:
    M, N = x.shape
    cutlass_dtype = _torch_to_cutlass_dtype[x.dtype]
    cfg = MXFP8QuantizeConfig(cutlass_dtype, fp8_dtype, rowwise=rowwise, colwise=colwise,
                               with_gemm_swizzled_scales=with_gemm_swizzled_scales,
                               with_amax=with_amax, activation=activation)
    fn_name = _get_compiled_kernel(cfg, M, N)
    return fn_name

def quantize_mxfp8_cutedsl(
    x: torch.Tensor,
    fp8_dtype: str = "e4m3",
    rowwise: bool = True,
    colwise: bool = False,
    with_gemm_swizzled_scales: bool = False,
    with_amax: bool = False,
    activation: str = None,
):
    quantizer = MXFP8Quantizer(
        fp8_dtype=_FP8_DTYPES[fp8_dtype],
        rowwise=rowwise,
        columnwise=colwise,
    )
    quantizer.internal = True
    if with_gemm_swizzled_scales:
        quantizer.optimize_for_gemm = True
    fn_name = get_quantize_mxfp8_cutedsl_func(
        x=x,
        rowwise=quantizer.rowwise_usage,
        colwise=quantizer.columnwise_usage,
        fp8_dtype="e4m3" if quantizer.dtype == tex.DType.kFloat8E4M3 else "e5m2",
        with_gemm_swizzled_scales=quantizer.optimize_for_gemm,
        with_amax=with_amax,
        activation=activation,
    )
    output = tex.quantize_with_func(x, quantizer, None, fn_name)
    return output


# ---------------------------------------------------------------------------
# Hybrid MXFP8 + NVFP4 quantization (pure-Python driver)
# ---------------------------------------------------------------------------
# The hybrid kernel emits MXFP8 in one direction and NVFP4 in the other, all
# from a single shared-memory read of the input tile. It needs 4 extra output
# buffers + a global encode scale, which the TE C++ `quantize_with_func` path
# (fixed 6-slot tensor arg list, no scalar support) can't carry. So this driver
# compiles the kernel directly and launches it via DLPack — no C++/rebuild, no
# tvm-ffi. Outputs use the simple compact, NON-swizzled scale layout (matches
# `NVFP4QuantizerRef`), not TE's production swizzled/padded layout.

_hybrid_compile_cache: dict = {}


def _get_compiled_hybrid_kernel(cfg, M, N):
    """Compile (and cache) the hybrid kernel for (cfg flags, M, N). The arg
    order MUST mirror MXFP8QuantizeSmemKernel.__call__; disabled slots compile
    as None so the runtime call lines up positionally."""
    key = (cfg.DTYPE.__name__, cfg.FP8_DTYPE,
           int(cfg.ROWWISE), int(cfg.COLWISE),
           int(cfg.NVFP4_ROWWISE), int(cfg.NVFP4_COLWISE), M, N)
    if key in _hybrid_compile_cache:
        return _hybrid_compile_cache[key]

    kw = dict(stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=16)

    def fake(dtype, shape):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, **kw)

    use_nvfp4 = cfg.NVFP4_ROWWISE or cfg.NVFP4_COLWISE
    args = [
        fake(cfg.DTYPE, (M, N)),                                          # mX
        fake(cute.Uint8, (M, N))               if cfg.ROWWISE else None,  # mO_row
        fake(cute.Uint8, (M, N // SCALE_DIM))  if cfg.ROWWISE else None,  # mS_row
        fake(cute.Uint8, (M, N))               if cfg.COLWISE else None,  # mO_col
        fake(cute.Uint8, (M // SCALE_DIM, N))  if cfg.COLWISE else None,  # mS_col
        None,                                                             # mAmax
        fake(cute.Uint8, (M, N // 2))                if cfg.NVFP4_ROWWISE else None,  # mO_nvfp4_row
        fake(cute.Uint8, (M, N // SCALE_DIM_NVFP4))  if cfg.NVFP4_ROWWISE else None,  # mS_nvfp4_row
        fake(cute.Uint8, (N, M // 2))                if cfg.NVFP4_COLWISE else None,  # mO_nvfp4_col
        fake(cute.Uint8, (N, M // SCALE_DIM_NVFP4))  if cfg.NVFP4_COLWISE else None,  # mS_nvfp4_col
        Float32(1.0) if use_nvfp4 else None,                              # s_enc (placeholder)
    ]
    # Compile the 11-param `forward_hybrid` entry (NOT `__call__`): the extra
    # NVFP4 params are free here (direct compile, no --enable-tvm-ffi wrapper).
    compiled = cute.compile[(GPUArch("sm_100a"),)](
        HybridQuantizeSmemKernel(cfg), *args
    )
    _hybrid_compile_cache[key] = compiled
    return compiled


def _hybrid_global_s_enc(x: torch.Tensor) -> float:
    """Global NVFP4 encode scale S_enc = 448*6 / global_amax, computed in fp32
    to bit-match NVFP4QuantizerRef (clamp to FLT_MAX; 0/inf amax -> 1.0)."""
    ga = x.abs().max().to(torch.float32)
    flt_max = torch.finfo(torch.float32).max
    s = torch.minimum(
        torch.tensor(448.0 * 6.0, dtype=torch.float32, device=x.device) / ga,
        torch.tensor(flt_max, dtype=torch.float32, device=x.device),
    )
    if float(ga) == 0.0 or float(s) == 0.0:
        return 1.0
    return float(s.item())


def quantize_hybrid_cutedsl(
    x: torch.Tensor,
    fp8_dtype: str = "e4m3",
    mxfp8_rowwise: bool = True,
    mxfp8_colwise: bool = False,
    nvfp4_rowwise: bool = False,
    nvfp4_colwise: bool = True,
):
    """Hybrid MXFP8 + NVFP4 quantization in one fused kernel.

    MXFP8 is produced in the rowwise/colwise direction(s) selected by
    `mxfp8_*`; NVFP4 in the direction(s) selected by `nvfp4_*` — reusing the
    same shared-memory input tile (single DRAM read). The canonical hybrid is
    `mxfp8_rowwise + nvfp4_colwise` (the default).

    Returns a dict of the enabled output torch.uint8 tensors:
      MXFP8:  rowwise_data (M,N), rowwise_scale (M,N/32) [E8M0]
              colwise_data (M,N), colwise_scale (M/32,N) [E8M0]
      NVFP4:  nvfp4_rowwise_data (M,N/2),  nvfp4_rowwise_scale (M,N/16) [E4M3]
              nvfp4_colwise_data (N,M/2),  nvfp4_colwise_scale (N,M/16) [E4M3] (transposed)
    plus "s_enc" (the global encode scale used).
    """
    assert x.is_cuda and x.is_contiguous() and x.ndim == 2, "expect a contiguous 2D CUDA tensor"
    assert x.dtype in (torch.bfloat16, torch.float16), \
        "hybrid PoC supports 16-bit input (bf16/fp16) only"
    M, N = x.shape
    assert M % (TILE_Y * NUM_TILES) == 0, f"M={M} must be a multiple of {TILE_Y * NUM_TILES}"
    assert N % TILE_X == 0, f"N={N} must be a multiple of {TILE_X}"
    assert mxfp8_rowwise or mxfp8_colwise or nvfp4_rowwise or nvfp4_colwise, \
        "no output direction enabled"

    cfg = MXFP8QuantizeConfig(
        _torch_to_cutlass_dtype[x.dtype], fp8_dtype,
        rowwise=mxfp8_rowwise, colwise=mxfp8_colwise,
        nvfp4_rowwise=nvfp4_rowwise, nvfp4_colwise=nvfp4_colwise,
    )
    use_nvfp4 = nvfp4_rowwise or nvfp4_colwise
    s_enc_val = _hybrid_global_s_enc(x) if use_nvfp4 else 1.0

    dev = x.device
    out = {}
    call = [cute.runtime.from_dlpack(x, assumed_align=16)]

    def add(name, shape, enabled):
        if enabled:
            t = torch.empty(shape, dtype=torch.uint8, device=dev)
            out[name] = t
            call.append(cute.runtime.from_dlpack(t, assumed_align=16))
        else:
            call.append(None)

    # Order MUST match _get_compiled_hybrid_kernel / __call__.
    add("rowwise_data",  (M, N),               cfg.ROWWISE)
    add("rowwise_scale", (M, N // SCALE_DIM),  cfg.ROWWISE)
    add("colwise_data",  (M, N),               cfg.COLWISE)
    add("colwise_scale", (M // SCALE_DIM, N),  cfg.COLWISE)
    call.append(None)                                                    # mAmax
    add("nvfp4_rowwise_data",  (M, N // 2),                cfg.NVFP4_ROWWISE)
    add("nvfp4_rowwise_scale", (M, N // SCALE_DIM_NVFP4),  cfg.NVFP4_ROWWISE)
    add("nvfp4_colwise_data",  (N, M // 2),                cfg.NVFP4_COLWISE)
    add("nvfp4_colwise_scale", (N, M // SCALE_DIM_NVFP4),  cfg.NVFP4_COLWISE)
    call.append(Float32(s_enc_val) if use_nvfp4 else None)               # s_enc

    compiled = _get_compiled_hybrid_kernel(cfg, M, N)
    compiled(*call)
    out["s_enc"] = s_enc_val
    return out
