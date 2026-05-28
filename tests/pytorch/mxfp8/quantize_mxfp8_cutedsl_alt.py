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
# Pin CuTeDSL compile target to the current device's Blackwell SM. Must be set
# before cutlass imports so env detection in base_dsl picks it up; also passed
# explicitly below.
def _detect_cute_dsl_arch() -> str:
    try:
        import torch as _torch
        major, minor = _torch.cuda.get_device_capability(_torch.cuda.current_device())
    except Exception:
        return "sm_100a"
    # Map known Blackwell capabilities to their CuTeDSL arch-specific targets.
    # Fall back to the plain (non-"a") target if the arch-specific one is unknown.
    return f"sm_{major}{minor}"

os.environ.setdefault("CUTE_DSL_ARCH", _detect_cute_dsl_arch())

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

# Rowwise thread layout (used by IS_DBIAS path B):
#   THREADS_X = TILE_X / SCALE_DIM     (= 2 — two scale-blocks per row)
#   THREADS_Y = THREADS_PER_CHUNK / THREADS_X     (= 32 — one row per Y thread)
DBIAS_THREADS_X = TILE_X // SCALE_DIM
DBIAS_THREADS_Y = THREADS_PER_CHUNK // DBIAS_THREADS_X
# Width of the path-B shmem transpose buffer: 2 thread-X groups, each
# `SCALE_DIM` floats wide plus 1 padding slot so each tid_X group starts
# on a different bank to avoid conflicts during the per-column read.
DBIAS_BUFF_WIDTH = DBIAS_THREADS_X * (SCALE_DIM + 1)
DBIAS_BUFF_SIZE = DBIAS_THREADS_Y * DBIAS_BUFF_WIDTH

# ---------------------------------------------------------------------------
# Low-level DSL operations
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Kernel configuration
# ---------------------------------------------------------------------------
class MXFP8QuantizeConfig:
    def __init__(self, dtype, fp8_dtype="e4m3", rowwise=True, colwise=False,
                 with_gemm_swizzled_scales=False, with_amax=False,
                 activation=None, with_dbias=False, is_dact=False):
        self.DTYPE = dtype
        self.FP8_DTYPE = fp8_dtype
        self.ROWWISE = rowwise
        self.COLWISE = colwise
        self.WITH_GEMM_SWIZZLED_SCALES = with_gemm_swizzled_scales
        # When True, kernel reduces max(|x|) across the full tensor and atomic-
        # maxes into a caller-provided 1-element f32 (delayed-scaling FP8).
        self.WITH_AMAX = with_amax
        # Optional fused activation. When set, the kernel applies OP per
        # element before amax+cast. The "d"-prefixed entries are derivative
        # activations — these select the IS_DACT path which loads a second
        # input tensor (the saved forward `act_input`) and computes
        # `elt = grad · dOP(act_in)`. Plain entries select the IS_ACT path.
        if activation is not None and activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; expected one of "
                f"{sorted(_ACTIVATIONS)} or None")
        self.ACTIVATION = activation
        # Derived flag — fully determined by the activation name. The kernel
        # entry point reads this to dispatch IS_DACT vs IS_ACT/none.
        self.IS_DACT = is_dact
        # IS_DBIAS — accumulate per-column sum of post-activation values into
        # a per-CTA workspace, then reduce externally to produce the bias
        # gradient. Two paths exist depending on whether colwise scaling is
        # enabled:
        #   path A (colwise=True): each thread owns a column, accumulator is
        #       a single Float32, written directly to workspace.
        #   path B (colwise=False, rowwise=True): each thread owns a row
        #       strip of 32 different columns; per-element thread_dbias array
        #       is shmem-transposed after the consumer loop so each thread
        #       ends up owning a column for the workspace write.
        self.WITH_DBIAS = with_dbias
        if with_dbias and not (rowwise or colwise):
            raise ValueError("with_dbias=True requires rowwise or colwise to be True")
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
        mActIn: cute.Tensor, # Forward activation output, only used in IS_DACT path
        mO_row: Optional[cute.Tensor], mS_row: Optional[cute.Tensor], # Rowwise output and scale tensors
        mO_col: Optional[cute.Tensor], mS_col: Optional[cute.Tensor], # Colwise output and scale tensors
        mNoop: cute.Tensor, # Skip flag — if *noop == 1.0, kernel exits immediately
        mAmax: cute.Tensor, # Global amax accumulator, only used in WITH_AMAX path
        mDbias: Optional[cute.Tensor], # Per-CTA-row partial dbias sums which is reduced down to (N,) by a separate kernel, only used in WITH_DBIAS path
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
        
        tma_atom_act = None
        tma_src_act = None
        if cutlass.const_expr(cfg.IS_DACT):
            # Second input descriptor — only consumed by the IS_DACT path. When
            # not is_dact, the wrapper aliases this onto x's data ptr so the
            # descriptor is well-formed but never read.
            tma_atom_act, tma_src_act = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_load, mActIn, smem_tile_layout, cta_tiler, num_multicast=1,
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
            mX, mO_row, mS_row, mO_col, mS_col, mNoop, mAmax, mDbias,
            max_norm_rcp, mX.element_type,
            tma_atom, tma_src,
            tma_atom_act, tma_src_act,
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
        mO_row,
        mS_row,
        mO_col,
        mS_col,
        mNoop, 
        mAmax, 
        mDbias,
        max_norm_rcp,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom, tma_src, # how to use TMA to copy the input
        tma_atom_act, tma_src_act,   # only used by the IS_DACT path
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

        # IS_DACT (backward activation) needs a paired G2S TMA load and a
        # different inner compute (`elt = grad · dOP(act_in)`), so it
        # gets its own top-level body — no nested const-expr branches
        # threaded through the producer/consumer of the IS_ACT/plain path.
        if cutlass.const_expr(cfg.IS_DACT):
            self._kernel_main_dact(
                mX, mO_row, mS_row, mO_col, mS_col, mAmax, mDbias,
                max_norm_rcp, dtype,
                tma_atom, tma_src,
                tma_atom_act, tma_src_act,
                tma_atom_out_row, tma_dst_out_row,
                tma_atom_out_col, tma_dst_out_col,
            )
        else:
            self._kernel_main(
                mX, mO_row, mS_row, mO_col, mS_col, mAmax, mDbias,
                max_norm_rcp, dtype,
                tma_atom, tma_src,
                tma_atom_out_row, tma_dst_out_row,
                tma_atom_out_col, tma_dst_out_col,
            )

    @cute.jit
    def _kernel_main(
        self,
        mX, mO_row, mS_row, mO_col, mS_col, mAmax, mDbias,
        max_norm_rcp,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom, tma_src,
        tma_atom_out_row, tma_dst_out_row,
        tma_atom_out_col, tma_dst_out_col,
    ):
        cfg = self.cfg

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
        elif cutlass.const_expr(cfg.ROWWISE and cfg.WITH_DBIAS and not cfg.COLWISE):
            # Path B: rowwise-only with dbias. Allocates an additional
            # `[THREADS_Y, DBIAS_BUFF_WIDTH]` f32 buffer for the post-loop
            # shmem transpose that aligns thread→column ownership before
            # the workspace write.
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sDbiasBuf: cute.struct.MemRange[Float32, DBIAS_BUFF_SIZE]
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
        # Per-thread, per-column dbias accumulator. Each thread owns column
        # `tidx` of the colwise pass. Threaded through `_process_colwise` so
        # extension is element-by-element across stages, matching TE's
        # cumulative-sum order bit-exactly. Always init to 0 (cheap dead
        # arg in the const-expr path where it's not used).
        block_dbias = Float32(0.0)
        # Path-B (rowwise-only + dbias) per-element accumulator: each thread
        # owns 32 different columns within its row strip; we keep one f32
        # register per element and shmem-transpose at the bottom of the
        # kernel so each thread ends up owning a single column for the
        # workspace write. List of length SCALE_DIM (= 32).
        thread_dbias_rw = [Float32(0.0) for _ in range(SCALE_DIM)]

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
                    mS_col_stage, max_norm_rcp, block_dbias,
                )




                # amax_c, block_dbias = self._process_colwise(
                #     sX_tile, sO_col_tile,
                #     mS_col_stage, max_norm_rcp, block_dbias,
                # )
                # if cutlass.const_expr(cfg.WITH_AMAX):
                #     block_amax = cute.arch.fmax(block_amax, amax_c)
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
                # amax_r, thread_dbias_rw = self._process_rowwise(
                amax_r, thread_dbias_rw = self._process_rowwise(
                    sX_tile, sO_row_tile,
                    mS_row_stage, max_norm_rcp, thread_dbias_rw,
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

        # ---- dbias workspace writeback ------------------------------------
        # Path A (colwise+dbias): each thread owns column `bidx*TILE_X + tidx`
        # of this CTA's tile and just writes its accumulator. No barrier.
        # Path B (rowwise-only+dbias): each thread owns 32 different columns
        # (a row strip), so we need a shmem transpose first — write each
        # thread's per-element accumulators to sDbiasBuf, sync, then each
        # thread reads back one column's worth of values across THREADS_Y
        # rows and sums them.
        if cutlass.const_expr(cfg.WITH_DBIAS and cfg.COLWISE):
            mDbias[bidy, bidx * TILE_X + tidx] = block_dbias
        elif cutlass.const_expr(cfg.WITH_DBIAS and not cfg.COLWISE):
            self._dbias_path_b_writeback(
                storage.sDbiasBuf, mDbias, thread_dbias_rw, bidx, bidy, tidx,
            )

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
    def _kernel_main_dact(
        self,
        mX, mO_row, mS_row, mO_col, mS_col, mAmax, mDbias,
        max_norm_rcp,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom, tma_src,
        tma_atom_act, tma_src_act,
        tma_atom_out_row, tma_dst_out_row,
        tma_atom_out_col, tma_dst_out_col,
    ):
        """IS_DACT (backward activation) variant of `_kernel_main`.

        Differs from the non-DACT body in three structural places, each
        kept top-level (no nested const-expr through one shared loop):
          - SharedStorage adds `sX_act` for the saved forward input.
          - Pipeline `tx_count = 2 * tile_bytes` since each stage issues
            two paired G2S copies that share one mbarrier.
          - Producer issues 2 `cute.copy` calls per stage; consumer reads
            from both `sX` (grad_y) and `sX_act` (forward `x`) and applies
            `elt = grad_y · dOP(x)` via `_process_*_dact`.
        """
        cfg = self.cfg

        # SharedStorage: same shape as non-DACT plus `sX_act`.
        if cutlass.const_expr(cfg.ROWWISE and cfg.COLWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sX_act: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        elif cutlass.const_expr(cfg.ROWWISE and cfg.WITH_DBIAS and not cfg.COLWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sX_act: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sDbiasBuf: cute.struct.MemRange[Float32, DBIAS_BUFF_SIZE]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        elif cutlass.const_expr(cfg.ROWWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sX_act: cute.struct.Align[
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
                sX_act: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        sX = storage.sX.get_tensor(
            cute.make_layout(
                ((TILE_Y, TILE_X), NUM_STAGES),
                stride=((TILE_X, 1), TILE_Y * TILE_X),
            )
        )
        sX_act = storage.sX_act.get_tensor(
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

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_act)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        num_warps = THREADS_PER_CHUNK // 32
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_warps)

        # Doubled tx_count — each stage's mbarrier expects bytes from BOTH
        # cute.copy ops (grad and act_in). The two TMAs share one barrier
        # and individually contribute their bytes via the hardware path.
        tx_count = 2 * (TILE_Y * TILE_X * dtype.width // 8)

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_storage.data_ptr(),
            num_stages=NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=None,
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

        gX_tiled = cute.zipped_divide(tma_src, (TILE_Y, TILE_X))
        gActIn_tiled = cute.zipped_divide(tma_src_act, (TILE_Y, TILE_X))

        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom, 0, cute.make_layout(1), sX, gX_tiled,
        )
        tXsX_act, tXgActIn = cute.nvgpu.cpasync.tma_partition(
            tma_atom_act, 0, cute.make_layout(1), sX_act, gActIn_tiled,
        )

        if cutlass.const_expr(cfg.ROWWISE):
            gO_row_tiled = cute.zipped_divide(tma_dst_out_row, (TILE_Y, TILE_X))
            tXsO_row, tXgO_row = cute.nvgpu.cpasync.tma_partition(
                tma_atom_out_row, 0, cute.make_layout(1), sO_row, gO_row_tiled,
            )
        if cutlass.const_expr(cfg.COLWISE):
            gO_col_tiled = cute.zipped_divide(tma_dst_out_col, (TILE_Y, TILE_X))
            tXsO_col, tXgO_col = cute.nvgpu.cpasync.tma_partition(
                tma_atom_out_col, 0, cute.make_layout(1), sO_col, gO_col_tiled,
            )

        cute.arch.sync_threads()

        # ---- Producer: warp 0 issues TWO TMA copies per tile, sharing the
        # same barrier. With tx_count = 2 * tile_bytes, the barrier is
        # satisfied only when both copies have completed.
        if warp_idx == 0:
            for stage in cutlass.range(num_tiles, unroll=1):
                mainloop_pipeline.producer_acquire(prod_state)
                tile_y = bidy * NUM_TILES + stage
                barrier = mainloop_pipeline.producer_get_barrier(prod_state)
                cute.copy(
                    tma_atom,
                    tXgX[(None, (tile_y, bidx))],
                    tXsX[(None, prod_state.index)],
                    tma_bar_ptr=barrier,
                )
                cute.copy(
                    tma_atom_act,
                    tXgActIn[(None, (tile_y, bidx))],
                    tXsX_act[(None, prod_state.index)],
                    tma_bar_ptr=barrier,
                )
                mainloop_pipeline.producer_commit(prod_state)
                prod_state.advance()

        if cutlass.const_expr(cfg.WITH_AMAX):
            block_amax = Float32(0.0)
        # See _kernel_main: threaded through `_process_colwise_dact` so the
        # cross-stage sum order matches TE's flat `partial_dbias += elt`.
        block_dbias = Float32(0.0)
        # Path-B per-element rowwise dbias accumulator (see _kernel_main).
        thread_dbias_rw = [Float32(0.0) for _ in range(SCALE_DIM)]

        # ---- Consumer: process each completed tile, reading both inputs.
        for stage in cutlass.range(num_tiles, unroll=1):
            mainloop_pipeline.consumer_wait(cons_state)
            sX_tile = sX[(None, cons_state.index)]
            sX_act_tile = sX_act[(None, cons_state.index)]
            base_row = (bidy * NUM_TILES + stage) * TILE_Y

            if cutlass.const_expr(cfg.COLWISE):
                sO_col_tile = sO_col[(None, cons_state.index)]
                amax_c, block_dbias = self._process_colwise_dact(
                    sX_tile, sX_act_tile, sO_col_tile,
                    base_row, bidx, tidx, mS_col, max_norm_rcp,
                    block_dbias,
                )
                if cutlass.const_expr(cfg.WITH_AMAX):
                    block_amax = cute.arch.fmax(block_amax, amax_c)
            if cutlass.const_expr(cfg.ROWWISE):
                sO_row_tile = sO_row[(None, cons_state.index)]
                amax_r, thread_dbias_rw = self._process_rowwise_dact(
                    sX_tile, sX_act_tile, sO_row_tile,
                    base_row, bidx, tidx, mS_row, max_norm_rcp,
                    thread_dbias_rw,
                )
                if cutlass.const_expr(cfg.WITH_AMAX):
                    block_amax = cute.arch.fmax(block_amax, amax_r)

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
                        tXsO_row[(None, cons_state.index)],
                        tXgO_row[(None, (tile_y, bidx))],
                    )
                if cutlass.const_expr(cfg.COLWISE):
                    cute.copy(
                        tma_atom_out_col,
                        tXsO_col[(None, cons_state.index)],
                        tXgO_col[(None, (tile_y, bidx))],
                    )
                cute.arch.cp_async_bulk_commit_group()

            mainloop_pipeline.consumer_release(cons_state)
            cons_state.advance()

        cute.arch.cp_async_bulk_wait_group(0, read=False)

        # ---- dbias workspace writeback (mirrors _kernel_main) -------------
        if cutlass.const_expr(cfg.WITH_DBIAS and cfg.COLWISE):
            mDbias[bidy, bidx * TILE_X + tidx] = block_dbias
        elif cutlass.const_expr(cfg.WITH_DBIAS and not cfg.COLWISE):
            self._dbias_path_b_writeback(
                storage.sDbiasBuf, mDbias, thread_dbias_rw, bidx, bidy, tidx,
            )

        # Same amax epilogue as _kernel_main — fmax over warps then atomic.
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
        thread_dbias_in,  # list[Float32] of length SCALE_DIM — per-element
                          # rowwise dbias accumulator. Only used when
                          # (cfg.WITH_DBIAS and not cfg.COLWISE). Caller passes it through
                          # untouched in non-path-B configs.
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
            thread_dbias_in,
            ACTIVATION=cfg.ACTIVATION,
            WITH_DBIAS=cfg.WITH_DBIAS,
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
        partial_dbias_in,  # Float32 — running per-thread, per-column dbias
                           # accumulator. We extend it one element at a time
                           # so cross-stage sum order matches TE bit-exactly.
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
            partial_dbias_in,
            ACTIVATION=cfg.ACTIVATION,
            DTYPE=cfg.DTYPE,
            FP8_DTYPE=cfg.FP8_DTYPE,
            WITH_DBIAS=cfg.WITH_DBIAS,
            SWIZZLE=cfg.WITH_GEMM_SWIZZLED_SCALES,
            TILE_X=TILE_X,
            TILE_Y=TILE_Y,
            SCALE_DIM=SCALE_DIM,
        )

    @cute.jit
    def _process_colwise_dact(
        self,
        sX_tile,         # (TILE_Y, TILE_X) — grad_y in IType
        sX_act_tile,     # (TILE_Y, TILE_X) — saved forward `x` in IType
        sO_col_tile,     # (TILE_Y, TILE_X) uint8 colwise FP8 output
        base_row, bidx, tidx,
        mS_col, max_norm_rcp,
        partial_dbias_in,  # Float32 — running dbias accumulator (see _process_colwise)
    ):
        """IS_DACT colwise pass: `elt = grad_y · dOP(act_in)`. Always takes
        the scalar f32 path (the bf16/fp16 fast path doesn't apply once we
        need to evaluate dOP in f32). Mirrors `_process_colwise` for
        scaling/cast/amax bookkeeping."""
        cfg = self.cfg
        block_off_X = bidx * TILE_X
        col_global = block_off_X + tidx

        sX_flat = cute.make_tensor(
            sX_tile.iterator,
            cute.make_layout((TILE_Y, TILE_X), stride=(TILE_X, 1)),
        )
        sX_act_flat = cute.make_tensor(
            sX_act_tile.iterator,
            cute.make_layout((TILE_Y, TILE_X), stride=(TILE_X, 1)),
        )
        sO_col_flat = cute.make_tensor(
            sO_col_tile.iterator,
            cute.make_layout((TILE_Y, TILE_X), stride=(TILE_X, 1)),
        )

        op = _ACTIVATIONS[cfg.ACTIVATION]   # the derivative function
        # Load both inputs as f32, compute elt = grad · dOP(act_in), and
        # interleave the dbias accumulation in the same per-element loop so
        # the f32 `partial = partial + grad·dOP(act)` chain fuses into a
        # single FFMA — matches TE's `elt *= OP; partial += elt;` pattern.
        # Defer IType-truncation until after dbias is accumulated.
        in_c = []
        kit = _packed16_kit(cfg.DTYPE) if _is_packed16(cfg.DTYPE) else None
        partial_dbias = partial_dbias_in
        for i in cutlass.range_constexpr(SCALE_DIM):
            grad = Float32(sX_flat[i, tidx])
            actin = Float32(sX_act_flat[i, tidx])
            elt = grad * op(actin)
            if cutlass.const_expr(cfg.WITH_DBIAS):
                partial_dbias = partial_dbias + elt
            in_c.append(elt)

        if cutlass.const_expr(_is_packed16(cfg.DTYPE)):
            for i in cutlass.range_constexpr(SCALE_DIM):
                in_c[i] = kit.truncate_f32(in_c[i])

        amax_c = Float32(0.0)
        for i in cutlass.range_constexpr(SCALE_DIM):
            amax_c = cute.arch.fmax(amax_c, fabs_f32(in_c[i]))

        # E8M0 scale → gmem (mS_col layout encodes swizzle if requested).
        biased_exp_c = float_to_e8m0(amax_c * max_norm_rcp)
        scale_row = base_row // SCALE_DIM
        mS_col[scale_row, col_global] = Uint8(biased_exp_c)

        # Scale + FP8 cast → smem.
        inv_scale_c = exp2f_rcp(biased_exp_c)
        cvt_to_fp8 = _cvt_f32_to_fp8(cfg.FP8_DTYPE)
        for i in cutlass.range_constexpr(SCALE_DIM):
            sO_col_flat[i, tidx] = Uint8(cvt_to_fp8(in_c[i] * inv_scale_c))

        return amax_c, partial_dbias

    @cute.jit
    def _process_rowwise_dact(
        self,
        sX_tile,
        sX_act_tile,
        sO_row_tile,
        base_row, bidx, tidx,
        mS_row, max_norm_rcp,
        thread_dbias_in,  # see _process_rowwise: per-element rowwise dbias accumulator
    ):
        """IS_DACT rowwise pass: `elt = grad_y · dOP(act_in)`. Same thread
        layout / wave / bank-swizzle as `_process_rowwise`, scalar f32 path.
        Path-B IS_DBIAS extends `thread_dbias_in` with each pre-truncation
        elt before the IType round-trip.
        """
        cfg = self.cfg
        thread_dbias = thread_dbias_in

        rowwise_thread_layout = cute.make_layout((TILE_Y, 2), stride=(2, 1))
        tid_Y, tid_X = rowwise_thread_layout.get_flat_coord(tidx)
        bank_group = (tidx % THREADS_PER_WARP) // THREADS_PER_BANK

        global_row = base_row + tid_Y
        scale_col = bidx * 2 + tid_X
        col_base_local = tid_X * SCALE_DIM

        sO_u32_ptr = cute.recast_ptr(sO_row_tile.iterator, dtype=Uint32)
        sO_u32 = cute.make_tensor(
            sO_u32_ptr,
            cute.make_layout(
                (TILE_Y, TILE_X // 4), stride=(TILE_X // 4, 1),
            ),
        )

        sX_rw = cute.make_tensor(
            sX_tile.iterator,
            cute.make_layout(
                (TILE_Y, 2, SCALE_DIM),
                stride=(TILE_X, SCALE_DIM, 1),
            ),
        )
        sX_act_rw = cute.make_tensor(
            sX_act_tile.iterator,
            cute.make_layout(
                (TILE_Y, 2, SCALE_DIM),
                stride=(TILE_X, SCALE_DIM, 1),
            ),
        )

        op = _ACTIVATIONS[cfg.ACTIVATION]
        kit = _packed16_kit(cfg.DTYPE) if _is_packed16(cfg.DTYPE) else None

        # Two-pass: first compute pre-truncation elt and (path B) accumulate
        # dbias from it; then truncate for amax/cast bookkeeping.
        in_r = [[None] * PACK_SIZE for _ in range(WAVES)]
        for w in cutlass.range_constexpr(WAVES):
            swz = ((w + bank_group) * PACK_SIZE) % SCALE_DIM
            for e in cutlass.range_constexpr(PACK_SIZE):
                grad = Float32(sX_rw[tid_Y, tid_X, swz + e])
                actin = Float32(sX_act_rw[tid_Y, tid_X, swz + e])
                in_r[w][e] = grad * op(actin)

        if cutlass.const_expr(cfg.WITH_DBIAS and not cfg.COLWISE):
            for w in cutlass.range_constexpr(WAVES):
                for e in cutlass.range_constexpr(PACK_SIZE):
                    j = w * PACK_SIZE + e
                    thread_dbias[j] = thread_dbias[j] + in_r[w][e]

        if cutlass.const_expr(_is_packed16(cfg.DTYPE)):
            for w in cutlass.range_constexpr(WAVES):
                for e in cutlass.range_constexpr(PACK_SIZE):
                    in_r[w][e] = kit.truncate_f32(in_r[w][e])

        amax_r = Float32(0.0)
        for w in cutlass.range_constexpr(WAVES):
            for e in cutlass.range_constexpr(PACK_SIZE):
                amax_r = cute.arch.fmax(amax_r, fabs_f32(in_r[w][e]))

        biased_exp_r = float_to_e8m0(amax_r * max_norm_rcp)
        mS_row[global_row, scale_col] = Uint8(biased_exp_r)

        inv_scale_r = exp2f_rcp(biased_exp_r)
        cvt_f32x2 = _cvt_f32x2_to_fp8x2(cfg.FP8_DTYPE)
        for w in cutlass.range_constexpr(WAVES):
            swz = ((w + bank_group) * PACK_SIZE) % SCALE_DIM
            v0 = in_r[w][0] * inv_scale_r
            v1 = in_r[w][1] * inv_scale_r
            v2 = in_r[w][2] * inv_scale_r
            v3 = in_r[w][3] * inv_scale_r
            p01 = cvt_f32x2(v1, v0)
            p23 = cvt_f32x2(v3, v2)
            quad = (p23 << Int32(16)) | p01
            sO_u32[tid_Y, (col_base_local + swz) // 4] = Uint32(quad)

        return amax_r, thread_dbias

    @cute.jit
    def _dbias_path_b_writeback(
        self,
        sDbiasBufStorage,   # struct.MemRange[Float32, DBIAS_BUFF_SIZE]
        mDbias,             # gmem workspace [blocks_Y, N] f32
        thread_dbias_rw,    # list[Float32] of length SCALE_DIM (per-element accum)
        bidx, bidy, tidx,
    ):
        """Path-B IS_DBIAS writeback: shmem-transpose per-element rowwise
        accumulators into per-column sums, then write each thread's column
        sum to the gmem workspace. Mirrors C++'s
        `cast/mxfp8/quantize_mxfp8.cuh:528-557`.

        Each thread (tid_Y, tid_X) holds 32 floats (`thread_dbias_rw[j]`,
        j ∈ [0, SCALE_DIM)) — one for each element of its row strip. We:
          1. Write the 32 floats to a `[THREADS_Y, DBIAS_BUFF_WIDTH]` shmem
             buffer at `[tid_Y, tid_X*(SCALE_DIM+1) + swizzled_idx + e]`.
             The +1 padding per tid_X group keeps the per-column read in
             different banks.
          2. sync_threads.
          3. Each thread reads one column down THREADS_Y rows and sums.
          4. Write to mDbias[bidy, bidx*TILE_X + tidx].
        """
        # Layout of the per-thread write region inside sDbiasBuf:
        #   row stride = DBIAS_BUFF_WIDTH (= THREADS_X * (SCALE_DIM + 1))
        #   tid_X group stride = SCALE_DIM + 1
        rowwise_thread_layout = cute.make_layout((TILE_Y, 2), stride=(2, 1))
        tid_Y, tid_X = rowwise_thread_layout.get_flat_coord(tidx)
        bank_group = (tidx % THREADS_PER_WARP) // THREADS_PER_BANK

        sDbiasBuf = sDbiasBufStorage.get_tensor(
            cute.make_layout(DBIAS_BUFF_SIZE),
        )

        shmem_thread_offset = tid_Y * DBIAS_BUFF_WIDTH + tid_X * (SCALE_DIM + 1)
        for w in cutlass.range_constexpr(WAVES):
            swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM
            swizzled_group_offset = shmem_thread_offset + swizzled_group_idx
            for e in cutlass.range_constexpr(PACK_SIZE):
                j = w * PACK_SIZE + e
                sDbiasBuf[swizzled_group_offset + e] = thread_dbias_rw[j]

        cute.arch.sync_threads()

        # Per-thread column read. Column index = tidx + scaling_block, where
        # scaling_block = tidx // SCALE_DIM accounts for the +1 padding gap
        # between tid_X groups (column 32 is padding when tidx >= 32).
        scaling_block = tidx // SCALE_DIM
        col_in_buf = tidx + scaling_block
        col_acc = Float32(0.0)
        for i in cutlass.range_constexpr(DBIAS_THREADS_Y):
            col_acc = col_acc + sDbiasBuf[i * DBIAS_BUFF_WIDTH + col_in_buf]

        mDbias[bidy, bidx * TILE_X + tidx] = col_acc


def _cfg_to_fn_name(cfg, M, N) -> str:
    """Deterministic registry key from (cfg, shape)."""
    key = (cfg.DTYPE.__name__, cfg.FP8_DTYPE,
           int(cfg.ROWWISE), int(cfg.COLWISE),
           int(cfg.WITH_GEMM_SWIZZLED_SCALES), int(cfg.WITH_AMAX),
           cfg.ACTIVATION or "none",
           int(cfg.WITH_DBIAS), int(cfg.IS_DACT),
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
    single_fake    = fake(Float32,    (1,),      kw_rm4_1d)
    out_row_fake   = fake(cute.Uint8, (M, N),    kw_rm16_2d) if cfg.ROWWISE   else None
    scale_row_fake = fake(cute.Uint8, SCALE_R,   kw_rm16_2d) if cfg.ROWWISE   else None
    out_col_fake   = fake(cute.Uint8, (M, N),    kw_rm16_2d) if cfg.COLWISE   else None
    scale_col_fake = fake(cute.Uint8, SCALE_C,   kw_rm16_2d) if cfg.COLWISE   else None
    ws_fake        = fake(Float32,    (WS_M, N), kw_rm4_2d)  if cfg.WITH_DBIAS else None

    compiled = cute.compile(
        kernel_obj,
        in_fake,                            # mX
        in_fake,                            # mActIn (aliased to mX unless IS_DACT)
        out_row_fake,   scale_row_fake,     # mO_row, mS_row
        out_col_fake,   scale_col_fake,     # mO_col, mS_col
        None,                        # mNoop
        None,                        # mAmax
        None,                            # mDbias
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

noop_flag = torch.tensor(1.0, dtype=torch.float32, device="cuda")  # if noop_flag[0] == 1.0 at launch, kernel returns immediately

def _tvm_ffi_dummy(shape, dtype, device):
    """Reusable empty buffer for tvm-ffi call-time shape checks on inactive
    kernel slots (unused direction outputs, unused dbias workspace). The
    kernel never reads/writes these (const-expr gated), so contents don't
    matter — only shape/dtype/device must match the compile-time fake-ptr."""
    key = (tuple(shape), dtype, str(device))
    buf = _TVM_FFI_DUMMY_CACHE.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _TVM_FFI_DUMMY_CACHE[key] = buf
    return buf


def get_quantize_mxfp8_cutedsl_func(
    x: torch.Tensor,
    quantized_output,
    fp8_dtype: str = "e4m3",
    rowwise: bool = True,
    colwise: bool = False,
    with_gemm_swizzled_scales: bool = False,
    noop: torch.Tensor = None,
    with_amax: bool = False,
    activation: str = None,
    act_input: torch.Tensor = None,
    compute_dbias: bool = False,
    is_dact: bool = False,
    do_nothing = False
) -> str:
    M, N = x.shape
    cutlass_dtype = _torch_to_cutlass_dtype[x.dtype]
    cfg = MXFP8QuantizeConfig(cutlass_dtype, fp8_dtype, rowwise=rowwise, colwise=colwise,
                               with_gemm_swizzled_scales=with_gemm_swizzled_scales,
                               with_amax=with_amax, activation=activation,
                               with_dbias=compute_dbias, is_dact=is_dact)
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
    act_input: torch.Tensor = None,
    compute_dbias: bool = False,
    is_dact: bool = False,
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
        quantized_output=tex.prepare_quantize(x, quantizer),
        rowwise=quantizer.rowwise_usage,
        colwise=quantizer.columnwise_usage,
        fp8_dtype="e4m3" if quantizer.dtype == tex.DType.kFloat8E4M3 else "e5m2",
        with_gemm_swizzled_scales=quantizer.optimize_for_gemm,
        with_amax=with_amax,
        activation=activation,
        act_input=act_input,
        compute_dbias=compute_dbias,
        is_dact=is_dact,
    )
    output = tex.quantize_with_func(x, quantizer, None, fn_name)
    return output
    