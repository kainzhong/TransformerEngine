# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 dequantization kernel implemented in CuTeDSL.

Faithful port of dequantize_mxfp8.cuh: given a 2D MXFP8 tensor (FP8E4M3/E5M2
data + E8M0 per-block scales), dequantize back to a higher-precision tensor
(BF16 / FP16 / FP32).

Matches the C++ kernel's tile dimensions and thread layout:
  CHUNK_DIM_Y = 128, CHUNK_DIM_X = 128, THREADS_PER_CHUNK = 128
  BUFFER_DIM_Y = 16,  BUFFER_DIM_X = 128, BUFFERS_NUM = 2
  ELEMS_PER_THREAD = 16, ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y = 8
  MXFP8 scale block = 32 elements

Grid: (ceil(N / 128), ceil(M / 128)). Each block processes one 128x128 chunk
in ITERATIONS=8 stages of 16x128 tiles, double-buffered in shared memory. The
C++ pipeline (2 buffers, 8 iterations) maps onto CuTeDSL's PipelineTmaAsync
(NUM_STAGES=2, NUM_TILES=8, one 16x128 tile per stage).
"""

# Local @cute.struct classes are SMEM-layout descriptors that need no docstrings.
# pylint: disable=missing-class-docstring

import logging
import os
from typing import Type

import cutlass
from cutlass import cute
from cutlass import pipeline
from cutlass import Float32, Int32
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

from transformer_engine.common.CuTeDSL.utils import (
    device_compute_capability,
    str_to_cutlass_dtype,
    exp2f,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import as_byte_tensor

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"

logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")

# Number of elements per MXFP8 scale block; they share one E8M0 scale factor.
MXFP8_BLOCK_SCALING_SIZE = 32
# How many threads are in one warp
THREADS_PER_WARP = 32

# Mirror the C++ constants (dequantize_mxfp8.cuh).
CHUNK_DIM_Y = 128
CHUNK_DIM_X = 128
THREADS_PER_CHUNK = 128
BUFFERS_NUM = 2
ELEMS_PER_THREAD = 16
BUFFER_DIM_Y = 16
BUFFER_DIM_X = CHUNK_DIM_X  # 128
ITERATIONS = CHUNK_DIM_Y // BUFFER_DIM_Y  # 8

THREADS_PER_CHUNK_X_ROWWISE = CHUNK_DIM_X // ELEMS_PER_THREAD  # 8 = 128 / 16
THREADS_PER_CHUNK_X_COLWISE = CHUNK_DIM_X  # 128
THREADS_PER_SCALE_X_ROWWISE = (
    MXFP8_BLOCK_SCALING_SIZE + ELEMS_PER_THREAD - 1
) // ELEMS_PER_THREAD  # 2 = ceil(32 / 16)

# GEMM-swizzled scale tile (matches swizzle.cuh).
GEMM_SWIZZLED_SCALE_TILE_DIM_X = 4
GEMM_SWIZZLED_SCALE_TILE_DIM_Y = 128


def gemm_swizzled_scale_idx(i: Int32, j: Int32, num_tiles_X: Int32) -> Int32:
    """Compact (i, j) scale index -> GEMM-swizzled flat scale index.

    Direct port of swizzle::gemm_swizzled_scale_idx (swizzle.cuh).
    """
    TILE_DIM_X = GEMM_SWIZZLED_SCALE_TILE_DIM_X
    TILE_DIM_Y = GEMM_SWIZZLED_SCALE_TILE_DIM_Y
    TILE_SIZE = TILE_DIM_X * TILE_DIM_Y
    tile_idx_X = j // TILE_DIM_X
    tile_idx_Y = i // TILE_DIM_Y
    idx_in_tile_X = j % TILE_DIM_X
    idx_in_tile_Y = i % TILE_DIM_Y
    idx = (tile_idx_Y * num_tiles_X + tile_idx_X) * TILE_SIZE
    idx += (idx_in_tile_Y % 32) * 16 + (idx_in_tile_Y // 32) * 4 + idx_in_tile_X
    return idx


def scale_flat_index(
    scale_offset_Y: Int32,
    scale_offset_X: Int32,
    scales_stride: Int32,
    num_scale_tiles_X: Int32,
    ROWWISE: cutlass.Constexpr,
    SWIZZLE: cutlass.Constexpr,
) -> Int32:
    """Flat offset of a scale byte, mirroring the C++ kernel's scale_idx computation.

    Compact layout is row-major (offset_Y * stride + offset_X). Swizzled rowwise
    keeps (Y, X); swizzled colwise swaps to (X, Y) -- exactly as the C++ kernel.
    """
    if cutlass.const_expr(SWIZZLE):
        if cutlass.const_expr(ROWWISE):
            return gemm_swizzled_scale_idx(scale_offset_Y, scale_offset_X, num_scale_tiles_X)
        return gemm_swizzled_scale_idx(scale_offset_X, scale_offset_Y, num_scale_tiles_X)
    return scale_offset_Y * scales_stride + scale_offset_X


@cute.jit
def dequantize_rowwise_mxfp8(
    sX_tile,  # (BUFFER_DIM_Y, BUFFER_DIM_X) fp8 smem view, post-TMA
    sO_tile,  # (BUFFER_DIM_Y, BUFFER_DIM_X) output smem view
    mS_flat,  # 1D uint8 view of the (padded) scale tensor
    scales_stride,  # Int32 — padded scale-tensor row stride
    num_scale_tiles_X,  # Int32 — number of 128-wide scale tiles along X (swizzle)
    scale_offset_Y_base,  # Int32 — global scale row of this tile's row 0
    scale_offset_X_base,  # Int32 — global scale col-block of this CTA's col 0
    OUT_DTYPE: cutlass.Constexpr,
    IN_DTYPE: cutlass.Constexpr,
    SWIZZLE: cutlass.Constexpr,
):
    """Dequantize one 16x128 tile rowwise (per-row 32-elt block scales).

    Thread layout mirrors the C++ kernel: THREADS_PER_CHUNK_X_ROWWISE (=8) threads
    per row, each owning ELEMS_PER_THREAD (=16) contiguous columns; two adjacent
    threads share one 32-element block scale.
    """
    tidx, _, _ = cute.arch.thread_idx()

    # (row, col-group) TV layout: 16 rows x 8 col-groups, 16 contiguous cols each.
    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout(
            (BUFFER_DIM_Y, THREADS_PER_CHUNK_X_ROWWISE),
            stride=(THREADS_PER_CHUNK_X_ROWWISE, 1),
        ),
        val_layout=cute.make_layout((1, ELEMS_PER_THREAD), stride=(0, 1)),
    )
    sX_thread = cute.composition(sX_tile, tv_layout)[tidx, None]  # (16,) fp8
    sO_thread = cute.composition(sO_tile, tv_layout)[tidx, None]  # (16,) OUT_DTYPE

    tid_rowwise_Y = tidx // THREADS_PER_CHUNK_X_ROWWISE
    tid_rowwise_X = tidx % THREADS_PER_CHUNK_X_ROWWISE

    scale_offset_Y = scale_offset_Y_base + tid_rowwise_Y
    scale_offset_X = scale_offset_X_base + tid_rowwise_X // THREADS_PER_SCALE_X_ROWWISE
    scale_idx = scale_flat_index(
        scale_offset_Y, scale_offset_X, scales_stride, num_scale_tiles_X, True, SWIZZLE
    )
    biased_exponent = Int32(mS_flat[scale_idx])
    block_scale = exp2f(biased_exponent)

    rIn = cute.make_rmem_tensor(ELEMS_PER_THREAD, IN_DTYPE)
    rOut = cute.make_rmem_tensor(ELEMS_PER_THREAD, OUT_DTYPE)
    cute.autovec_copy(sX_thread, rIn)
    for j in cutlass.range_constexpr(ELEMS_PER_THREAD):
        rOut[j] = OUT_DTYPE(block_scale * Float32(rIn[j]))
    cute.autovec_copy(rOut, sO_thread)


@cute.jit
def dequantize_colwise_mxfp8(
    sX_tile,  # (BUFFER_DIM_Y, BUFFER_DIM_X) fp8 smem view, post-TMA
    sO_tile,  # (BUFFER_DIM_Y, BUFFER_DIM_X) output smem view
    mS_flat,  # 1D uint8 view of the (padded) scale tensor
    scales_stride,  # Int32 — padded scale-tensor row stride
    num_scale_tiles_X,  # Int32 — number of 128-tall scale tiles along Y (swizzle)
    scale_offset_Y,  # Int32 — global scale row-block shared by this whole tile
    scale_offset_X_base,  # Int32 — global scale col of this CTA's col 0
    OUT_DTYPE: cutlass.Constexpr,
    IN_DTYPE: cutlass.Constexpr,
    SWIZZLE: cutlass.Constexpr,
):
    """Dequantize one 16x128 tile colwise (per-column 32-elt block scales).

    Each thread owns one column (THREADS_PER_CHUNK_X_COLWISE == BUFFER_DIM_X) and
    walks all BUFFER_DIM_Y rows; the whole tile shares one scale row-block.
    """
    tidx, _, _ = cute.arch.thread_idx()

    # Each thread owns one column: (1 x 128) threads, 16 rows of values per thread.
    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((1, THREADS_PER_CHUNK_X_COLWISE), stride=(BUFFER_DIM_X, 1)),
        val_layout=cute.make_layout((BUFFER_DIM_Y, 1), stride=(1, 1)),
    )
    sX_thread = cute.composition(sX_tile, tv_layout)[tidx, None]  # (16,) fp8
    sO_thread = cute.composition(sO_tile, tv_layout)[tidx, None]  # (16,) OUT_DTYPE

    scale_offset_X = scale_offset_X_base + tidx
    scale_idx = scale_flat_index(
        scale_offset_Y, scale_offset_X, scales_stride, num_scale_tiles_X, False, SWIZZLE
    )
    biased_exponent = Int32(mS_flat[scale_idx])
    block_scale = exp2f(biased_exponent)

    for i in cutlass.range_constexpr(BUFFER_DIM_Y):
        sO_thread[i] = OUT_DTYPE(block_scale * Float32(sX_thread[i]))


class MXFP8DequantizeConfig:
    """Config for a compiled CuTeDSL MXFP8 dequantize kernel. Fixed at compile time
    (behaves as const expressions inside the kernel)."""

    def __init__(
        self,
        out_dtype: str,
        fp8_dtype: str,
        rowwise: bool,
        colwise: bool,
        with_gemm_swizzled_scales: bool,
    ):
        if out_dtype is None or out_dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"unknown output dtype {out_dtype!r}; expected fp32|fp16|bf16")
        self.OUT_DTYPE = str_to_cutlass_dtype(out_dtype)
        self.OUT_DTYPE_STR = out_dtype
        if fp8_dtype not in ("e4m3", "e5m2"):
            raise ValueError(f"unknown FP8 dtype {fp8_dtype!r}; expected 'e4m3' or 'e5m2'")
        self.FP8_DTYPE = fp8_dtype
        self.IN_DTYPE = cutlass.Float8E4M3FN if fp8_dtype == "e4m3" else cutlass.Float8E5M2
        if not (rowwise or colwise):
            raise ValueError("at least one of rowwise or colwise must be true")
        # The C++ kernel dequantizes exactly one direction and prefers rowwise when
        # both are present (input_data = has_data() ? data : columnwise_data).
        self.ROWWISE = rowwise
        self.WITH_GEMM_SWIZZLED_SCALES = with_gemm_swizzled_scales

    def __str__(self):
        return (
            f"MXFP8DequantizeConfig(out_dtype={self.OUT_DTYPE_STR}, fp8_dtype={self.FP8_DTYPE}, "
            f"rowwise={self.ROWWISE}, swizzled={self.WITH_GEMM_SWIZZLED_SCALES})"
        )

    __repr__ = __str__


class MXFP8DequantizeKernel:
    """MXFP8 dequantize kernel that mirrors the CUDA C++ dequantize_mxfp8_kernel.

    `__call__` is the AOT-compiled entrypoint; `self` (hence `cfg`) is captured and
    fixed per compiled kernel.
    """

    _TILE_ROWS = BUFFER_DIM_Y  # 16
    _TILE_COLS = BUFFER_DIM_X  # 128
    _NUM_TILES = ITERATIONS  # 8 (one 128-row chunk per CTA)
    _NUM_STAGES = BUFFERS_NUM  # 2
    _THREADS_PER_CTA = THREADS_PER_CHUNK  # 128
    _NUM_WARPS = _THREADS_PER_CTA // THREADS_PER_WARP  # 4

    def __init__(self, cfg: MXFP8DequantizeConfig):
        self.cfg = cfg

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # FP8 input tensor to dequantize
        mO: cute.Tensor,  # Higher-precision output tensor
        mS: cute.Tensor,  # E8M0 per-block scales (rowwise or colwise, padded)
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(f"[CuTeDSL] MXFP8DequantizeKernel.__call__() with config: {self.cfg}\n")

        M = mX.shape[0]
        N = mX.shape[1]

        # Scales carry a native E8M0 dtype at the FFI boundary; work on raw bytes.
        mS = as_byte_tensor(mS)

        smem_tile_layout = cute.make_ordered_layout((self._TILE_ROWS, self._TILE_COLS), order=(1, 0))
        cta_tiler = (self._TILE_ROWS, self._TILE_COLS)

        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom, tma_src = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_load,
            mX,
            smem_tile_layout,
            cta_tiler,
            num_multicast=1,
        )
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_out, tma_dst_out = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_store,
            mO,
            smem_tile_layout,
            cta_tiler,
            num_multicast=1,
        )

        grid = [
            cute.ceil_div(Int32(N), self._TILE_COLS),
            cute.ceil_div(M, self._TILE_ROWS * self._NUM_TILES),
        ]
        block = [self._THREADS_PER_CTA]

        self.kernel(
            mX,
            mO,
            mS,
            mX.element_type,
            tma_atom,
            tma_src,
            tma_atom_out,
            tma_dst_out,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mX,
        mO,
        mS,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom,
        tma_src,  # Input TMA atoms
        tma_atom_out,
        tma_dst_out,  # Output TMA atoms
    ):
        """Device entry: dequantize one 128x128 chunk in 8 double-buffered 16x128 stages."""
        cfg = self.cfg

        @cute.struct
        class SharedStorage:
            mbar_storage: cute.struct.MemRange[cute.Int64, 2 * self._NUM_STAGES]
            sX: cute.struct.Align[
                cute.struct.MemRange[dtype, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES],
                128,
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[
                    cfg.OUT_DTYPE, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
                ],
                128,
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        tile_layout = cute.make_layout(
            ((self._TILE_ROWS, self._TILE_COLS), self._NUM_STAGES),
            stride=((self._TILE_COLS, 1), self._TILE_ROWS * self._TILE_COLS),
        )
        sX = storage.sX.get_tensor(tile_layout)
        sO = storage.sO.get_tensor(tile_layout)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        M = mX.shape[0]
        N = mX.shape[1]

        # Flat uint8 view of the (contiguous, padded) scale tensor; indexed by the
        # flat scale_idx exactly as scales_ptr[scale_idx] in the C++ kernel.
        scales_stride = Int32(mS.shape[1])
        mS_flat = cute.make_tensor(
            mS.iterator, cute.make_layout((mS.shape[0] * mS.shape[1],), stride=(1,))
        )
        # num_scale_tiles_X mirrors the C++ dispatcher: cols/128 rowwise, rows/128 colwise.
        if cutlass.const_expr(cfg.ROWWISE):
            num_scale_tiles_X = cute.ceil_div(Int32(N), 128)
        else:
            num_scale_tiles_X = cute.ceil_div(Int32(M), 128)

        # Only warp 0 is the producer (issues TMA); every warp is a consumer.
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self._NUM_WARPS)
        tx_count = self._TILE_ROWS * self._TILE_COLS * dtype.width // 8

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_storage.data_ptr(),
            num_stages=self._NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=None,  # single-CTA, no cluster/multicast
        )

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._NUM_STAGES
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._NUM_STAGES
        )

        # Valid 16-row tiles remaining from this CTA's 128-row chunk (M % 32 == 0, so
        # each 16-row tile is whole; OOB tiles are simply skipped -- TMA would drop
        # their stores anyway).
        num_tiles = cutlass.min(
            self._NUM_TILES,
            cute.ceil_div(M - bidy * self._TILE_ROWS * self._NUM_TILES, self._TILE_ROWS),
        )

        gX_tiled = cute.zipped_divide(tma_src, (self._TILE_ROWS, self._TILE_COLS))
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom, 0, cute.make_layout(1), sX, gX_tiled
        )
        gO_tiled = cute.zipped_divide(tma_dst_out, (self._TILE_ROWS, self._TILE_COLS))
        tXsO, tXgO = cute.nvgpu.cpasync.tma_partition(
            tma_atom_out, 0, cute.make_layout(1), sO, gO_tiled
        )

        cute.arch.sync_threads()

        # Prologue: warp 0 fills the pipeline with up to NUM_STAGES tiles.
        if warp_idx == 0:
            for s in cutlass.range_constexpr(self._NUM_STAGES):
                if s < num_tiles:
                    mainloop_pipeline.producer_acquire(prod_state)
                    tile_y = bidy * self._NUM_TILES + s
                    cute.copy(
                        tma_atom,
                        tXgX[(None, (tile_y, bidx))],
                        tXsX[(None, prod_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                    )
                    mainloop_pipeline.producer_commit(prod_state)
                    prod_state.advance()

        # Consumer loop: each warp dequantizes its tile, then refills the freed buffer.
        for tile_idx in cutlass.range(num_tiles, unroll=1):
            mainloop_pipeline.consumer_wait(cons_state)
            if warp_idx == 0:
                cute.arch.cp_async_bulk_wait_group(self._NUM_STAGES - 1, read=True)
            cute.arch.sync_threads()
            stage_idx = cons_state.index
            sX_tile = sX[(None, stage_idx)]
            sO_tile = sO[(None, stage_idx)]

            # tile_idx == the C++ kernel's `iter` within this CTA's 128-row chunk.
            if cutlass.const_expr(cfg.ROWWISE):
                # scales_rowwise_chunk_offset_Y + iter * BUFFER_DIM_Y
                scale_offset_Y_base = bidy * CHUNK_DIM_Y + tile_idx * BUFFER_DIM_Y
                # scales_rowwise_chunk_offset_X
                scale_offset_X_base = bidx * (CHUNK_DIM_X // MXFP8_BLOCK_SCALING_SIZE)
                dequantize_rowwise_mxfp8(
                    sX_tile,
                    sO_tile,
                    mS_flat,
                    scales_stride,
                    num_scale_tiles_X,
                    scale_offset_Y_base,
                    scale_offset_X_base,
                    cfg.OUT_DTYPE,
                    cfg.IN_DTYPE,
                    cfg.WITH_GEMM_SWIZZLED_SCALES,
                )
            else:
                # scales_colwise_chunk_offset_Y + (iter * BUFFER_DIM_Y) / SCALE_DIM_Y
                scale_offset_Y = bidy * (CHUNK_DIM_Y // MXFP8_BLOCK_SCALING_SIZE) + (
                    tile_idx * BUFFER_DIM_Y
                ) // MXFP8_BLOCK_SCALING_SIZE
                # scales_colwise_chunk_offset_X
                scale_offset_X_base = bidx * CHUNK_DIM_X
                dequantize_colwise_mxfp8(
                    sX_tile,
                    sO_tile,
                    mS_flat,
                    scales_stride,
                    num_scale_tiles_X,
                    scale_offset_Y,
                    scale_offset_X_base,
                    cfg.OUT_DTYPE,
                    cfg.IN_DTYPE,
                    cfg.WITH_GEMM_SWIZZLED_SCALES,
                )

            # Make the smem output writes visible to the TMA async proxy before store.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            mainloop_pipeline.consumer_release(cons_state)

            if warp_idx == 0:
                tile_y = bidy * self._NUM_TILES + tile_idx
                cute.copy(tma_atom_out, tXsO[(None, stage_idx)], tXgO[(None, (tile_y, bidx))])
                cute.arch.cp_async_bulk_commit_group()

            cons_state.advance()

            if warp_idx == 0:
                next_tile_idx = tile_idx + self._NUM_STAGES
                if next_tile_idx < num_tiles:
                    mainloop_pipeline.producer_acquire(prod_state)
                    tile_y = bidy * self._NUM_TILES + next_tile_idx
                    cute.copy(
                        tma_atom,
                        tXgX[(None, (tile_y, bidx))],
                        tXsX[(None, prod_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                    )
                    mainloop_pipeline.producer_commit(prod_state)
                    prod_state.advance()

        # Wait for in-flight TMA stores so the output is visible before the CTA exits.
        cute.arch.cp_async_bulk_wait_group(0, read=False)


def compile_cutedsl_function_from_cfg(cfg: MXFP8DequantizeConfig):
    """Return the compiled CuTeDSL function object for the given dequantize config."""
    # M, N divisible by the MXFP8 scale-block size (32), matching the C++ requirement.
    sym_M = cute.sym_int32(divisibility=MXFP8_BLOCK_SCALING_SIZE)
    sym_N = cute.sym_int32(divisibility=MXFP8_BLOCK_SCALING_SIZE)
    in_shape = out_shape = (sym_M, sym_N)
    # Padded scale-tensor extents (see MXFP8Quantizer::get_scale_shape):
    #   rowwise:    (roundup(M, 128),     roundup(N // 32, 4))
    #   columnwise: (roundup(M // 32, 4), roundup(N, 128))
    scale_rowwise_shape = (cute.sym_int32(divisibility=128), cute.sym_int32(divisibility=4))
    scale_colwise_shape = (cute.sym_int32(divisibility=4), cute.sym_int32(divisibility=128))
    scale_shape = scale_rowwise_shape if cfg.ROWWISE else scale_colwise_shape

    in_dtype = cfg.IN_DTYPE
    scale_dtype = cutlass.Float8E8M0FNU

    in_fake = cute.runtime.make_fake_compact_tensor(
        in_dtype, in_shape, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=16
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cfg.OUT_DTYPE,
        out_shape,
        stride_order=(1, 0),
        memspace=cute.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_fake = cute.runtime.make_fake_compact_tensor(
        scale_dtype,
        scale_shape,
        stride_order=(1, 0),
        memspace=cute.AddressSpace.gmem,
        assumed_align=4,
    )

    kernel_obj = MXFP8DequantizeKernel(cfg)
    compiled = cute.compile(
        kernel_obj,
        in_fake,  # mX
        out_fake,  # mO
        scale_fake,  # mS
        cute.runtime.make_fake_stream(),  # stream (tvm-ffi "handle" arg; C++ passes CUDA stream)
        options="--enable-tvm-ffi",
    )
    return compiled


def get_mxfp8_dequantization_function(
    fn_name: str,
    out_dtype: str,
    fp8_dtype: str,
    rowwise: bool,
    colwise: bool,
    with_gemm_swizzled_scales: bool,
) -> bool:
    """Compile the MXFP8 dequantize kernel for this config and register it in the TVM-FFI
    global registry under EXACTLY `fn_name`. Returns True on success (the C++ dispatcher
    then fetches it with GetGlobal(fn_name)); False if unsupported so the caller falls
    back to the CUDA C++ kernel.
    """
    if tvm_ffi.get_global_func(fn_name, allow_missing=True) is not None:
        return True

    major, minor = device_compute_capability()
    if major < 10:
        logger.warning(
            "CuTeDSL MXFP8 backend requires compute capability >= 10.0 (Blackwell), "
            "but detected %d.%d; falling back to the CUDA C++ kernel.",
            major,
            minor,
        )
        return False

    try:
        cfg = MXFP8DequantizeConfig(
            out_dtype=out_dtype,
            fp8_dtype=fp8_dtype,
            rowwise=rowwise,
            colwise=colwise,
            with_gemm_swizzled_scales=with_gemm_swizzled_scales,
        )
    except ValueError as e:
        logger.warning(
            "CuTeDSL MXFP8 dequantize backend does not support this config, "
            "falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False

    logger.debug("Compiling CuTeDSL MXFP8 dequantization kernel for %s", cfg)
    try:
        compiled = compile_cutedsl_function_from_cfg(cfg)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "CuTeDSL MXFP8 dequantize kernel compilation failed, "
            "falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False
    tvm_ffi.register_global_func(fn_name, compiled, override=True)

    return True
