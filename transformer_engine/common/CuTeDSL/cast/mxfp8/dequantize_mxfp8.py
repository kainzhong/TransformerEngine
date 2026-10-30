# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CuTeDSL MXFP8 dequantization kernel, with its compilation and TVM-FFI registration."""

# Local @cute.struct classes are shared-memory storage descriptors.
# pylint: disable=missing-class-docstring

import logging

import cutlass
from cutlass import cute
from cutlass import Float8E8M0FNU, Float32, Int32
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

from transformer_engine.common.CuTeDSL.utils import (
    device_compute_capability,
    exp2f,
    str_to_cutlass_dtype,
)
from .quantize_mxfp8_common import (
    CUTEDSL_DEBUG_LOGGING,
    MXFP8_BLOCK_SCALING_SIZE,
    SYM_N_DIVISIBILITY,
    derive_swizzled_scale_layout,
)

logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")


class MXFP8DequantizeConfig:
    """Compile-time configuration of the CuTeDSL MXFP8 dequantize kernel."""

    def __init__(
        self,
        fp8_dtype: str,
        out_dtype: str,
        rowwise: bool,
        with_gemm_swizzled_scales: bool,
    ):
        if fp8_dtype not in ("Float8E4M3", "Float8E5M2"):
            raise ValueError(
                f"unknown FP8 dtype {fp8_dtype!r}; expected 'Float8E4M3' or 'Float8E5M2'"
            )
        if out_dtype not in ("Float32", "Float16", "BFloat16"):
            raise ValueError(
                f"unknown output dtype {out_dtype!r}; expected Float32|Float16|BFloat16"
            )
        self.FP8_DTYPE = str_to_cutlass_dtype(fp8_dtype)
        self.FP8_DTYPE_STR = fp8_dtype
        self.OUT_DTYPE = str_to_cutlass_dtype(out_dtype)
        self.OUT_DTYPE_STR = out_dtype
        # Dequantize the rowwise data with rowwise scales, or else the columnwise data with
        # columnwise scales. Both data copies are (M, N) row-major.
        self.ROWWISE = rowwise
        self.WITH_GEMM_SWIZZLED_SCALES = with_gemm_swizzled_scales

    def __str__(self):
        return (
            f"MXFP8DequantizeConfig(fp8_dtype={self.FP8_DTYPE_STR}, "
            f"out_dtype={self.OUT_DTYPE_STR}, rowwise={self.ROWWISE}, "
            f"swizzled={self.WITH_GEMM_SWIZZLED_SCALES})"
        )

    __repr__ = __str__


class MXFP8DequantizeKernel:
    """MXFP8 dequantize with staged TMA I/O, a 1:1 port of dequantize_mxfp8_kernel in
    cast/mxfp8/dequantize_mxfp8.cuh (same tiling, thread mapping, barriers and pipelining).

    Each 128-thread CTA owns a (128, 128) chunk, processed as ITERATIONS (16, 128) stages through
    BUFFERS_NUM double-buffered input/output smem tiles. The next stage's TMA load is issued
    before the current one is consumed, and the output is written back with a TMA store.
    """

    _CHUNK_DIM_Y = 128
    _CHUNK_DIM_X = 128
    _THREADS_PER_CHUNK = 128
    _BUFFERS_NUM = 2
    _ELEMS_PER_THREAD = 16
    _BUFFER_DIM_Y = 16
    _BUFFER_DIM_X = _CHUNK_DIM_X
    _THREADS_PER_CHUNK_X_ROWWISE = _CHUNK_DIM_X // _ELEMS_PER_THREAD
    _ITERATIONS = _CHUNK_DIM_Y // _BUFFER_DIM_Y
    _THREADS_PER_SCALE_X_ROWWISE = MXFP8_BLOCK_SCALING_SIZE // _ELEMS_PER_THREAD

    def __init__(self, cfg: MXFP8DequantizeConfig):
        self.cfg = cfg

    @cute.jit
    def __call__(self, mX: cute.Tensor, mS: cute.Tensor, mO: cute.Tensor, stream: CUstream):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(f"[CuTeDSL] MXFP8DequantizeKernel.__call__() with config: {self.cfg}\n")

        M, N = mX.shape
        if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            if cutlass.const_expr(self.cfg.ROWWISE):
                mS, _ = derive_swizzled_scale_layout(M, N, True, False, mS, None)
            else:
                _, mS = derive_swizzled_scale_layout(M, N, False, True, None, mS)

        # Unswizzled (16, 128) boxes, like create_2D_tensor_map's default.
        buffer_tiler = (self._BUFFER_DIM_Y, self._BUFFER_DIM_X)
        smem_buffer_layout = cute.make_ordered_layout(buffer_tiler, order=(1, 0))
        tma_atom_in, tma_in = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), mX, smem_buffer_layout, buffer_tiler
        )
        tma_atom_out, tma_out = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), mO, smem_buffer_layout, buffer_tiler
        )

        grid = [
            cute.ceil_div(Int32(N), self._CHUNK_DIM_X),
            cute.ceil_div(M, self._CHUNK_DIM_Y),
            1,
        ]
        self.kernel(mS, tma_atom_in, tma_in, tma_atom_out, tma_out).launch(
            grid=grid, block=[self._THREADS_PER_CHUNK, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mS: cute.Tensor,
        tma_atom_in: cute.CopyAtom,
        tma_in: cute.Tensor,
        tma_atom_out: cute.CopyAtom,
        tma_out: cute.Tensor,
    ):
        """Device entry: out = exp2(scale - 127) * x, computed in FP32."""
        ROWWISE = self.cfg.ROWWISE
        BUFFER_SIZE = self._BUFFER_DIM_Y * self._BUFFER_DIM_X
        ITERATIONS = self._ITERATIONS
        BUFFERS_NUM = self._BUFFERS_NUM

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        is_master_thread = tidx == 0

        tid_rowwise_Y = tidx // self._THREADS_PER_CHUNK_X_ROWWISE
        tid_rowwise_X = tidx % self._THREADS_PER_CHUNK_X_ROWWISE
        tid_colwise_X = tidx

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cute.Int64, ITERATIONS]
            in_sh: cute.struct.Align[
                cute.struct.MemRange[self.cfg.FP8_DTYPE, BUFFERS_NUM * BUFFER_SIZE], 128
            ]
            out_sh: cute.struct.Align[
                cute.struct.MemRange[self.cfg.OUT_DTYPE, BUFFERS_NUM * BUFFER_SIZE], 128
            ]

        storage = cutlass.memory.SmemAllocator().allocate(SharedStorage)
        buffers_layout = cute.make_layout(
            ((self._BUFFER_DIM_Y, self._BUFFER_DIM_X), BUFFERS_NUM),
            stride=((self._BUFFER_DIM_X, 1), BUFFER_SIZE),
        )
        in_sh = storage.in_sh.get_tensor(buffers_layout)
        out_sh = storage.out_sh.get_tensor(buffers_layout)
        mbar = storage.mbar.data_ptr()

        tile_coord_x = bidx
        tile_coord_y0 = bidy * ITERATIONS
        buffer_tiler = (self._BUFFER_DIM_Y, self._BUFFER_DIM_X)
        tXs_in, tXg_in = cute.nvgpu.cpasync.tma_partition(
            tma_atom_in, 0, cute.make_layout(1), in_sh, cute.zipped_divide(tma_in, buffer_tiler)
        )
        tXs_out, tXg_out = cute.nvgpu.cpasync.tma_partition(
            tma_atom_out, 0, cute.make_layout(1), out_sh, cute.zipped_divide(tma_out, buffer_tiler)
        )
        transaction_size = BUFFER_SIZE * self.cfg.FP8_DTYPE.width // 8

        # All THREADS_PER_CHUNK threads arrive on every barrier, as in the CUDA kernel.
        if is_master_thread:
            for it in cutlass.range_constexpr(ITERATIONS):
                cute.arch.mbarrier_init(mbar + it, self._THREADS_PER_CHUNK)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # Prologue: load stage 0. The TMA copy elects one lane itself, so warp 0 issues it.
        if warp_idx == 0:
            cute.copy(
                tma_atom_in,
                tXg_in[(None, (tile_coord_y0, tile_coord_x))],
                tXs_in[(None, 0)],
                tma_bar_ptr=mbar,
            )
        if is_master_thread:
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, transaction_size)
        else:
            cute.arch.mbarrier_arrive(mbar)

        rIn = cute.make_rmem_tensor(cute.make_layout(self._ELEMS_PER_THREAD), self.cfg.FP8_DTYPE)
        rOut = cute.make_rmem_tensor(cute.make_layout(self._ELEMS_PER_THREAD), self.cfg.OUT_DTYPE)

        for it in cutlass.range_constexpr(ITERATIONS):
            buff = it % BUFFERS_NUM
            next_it = it + 1
            if cutlass.const_expr(next_it < ITERATIONS):
                if warp_idx == 0:
                    cute.copy(
                        tma_atom_in,
                        tXg_in[(None, (tile_coord_y0 + next_it, tile_coord_x))],
                        tXs_in[(None, next_it % BUFFERS_NUM)],
                        tma_bar_ptr=mbar + next_it,
                    )
                if is_master_thread:
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar + next_it, transaction_size)
                else:
                    cute.arch.mbarrier_arrive(mbar + next_it)

            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.mbarrier_wait(mbar + it, 0)

            buffer_layout = cute.make_ordered_layout(buffer_tiler, order=(1, 0))
            sIn = cute.make_tensor(in_sh[None, buff].iterator, buffer_layout)
            sOut = cute.make_tensor(out_sh[None, buff].iterator, buffer_layout)
            if cutlass.const_expr(ROWWISE):
                scale_offset_Y = bidy * self._CHUNK_DIM_Y + it * self._BUFFER_DIM_Y + tid_rowwise_Y
                scale_offset_X = (
                    bidx * (self._CHUNK_DIM_X // MXFP8_BLOCK_SCALING_SIZE)
                    + tid_rowwise_X // self._THREADS_PER_SCALE_X_ROWWISE
                )
                block_scale = exp2f(mS[scale_offset_Y, scale_offset_X])
                vec = (1, self._ELEMS_PER_THREAD)
                cute.autovec_copy(
                    cute.zipped_divide(sIn, vec)[None, (tid_rowwise_Y, tid_rowwise_X)], rIn
                )
                rOut.store((rIn.load().to(Float32) * block_scale).to(self.cfg.OUT_DTYPE))
                cute.autovec_copy(
                    rOut, cute.zipped_divide(sOut, vec)[None, (tid_rowwise_Y, tid_rowwise_X)]
                )
            else:
                scale_offset_Y = (
                    bidy * (self._CHUNK_DIM_Y // MXFP8_BLOCK_SCALING_SIZE)
                    + (it * self._BUFFER_DIM_Y) // MXFP8_BLOCK_SCALING_SIZE
                )
                scale_offset_X = bidx * self._CHUNK_DIM_X + tid_colwise_X
                block_scale = exp2f(mS[scale_offset_Y, scale_offset_X])
                for i in cutlass.range_constexpr(self._BUFFER_DIM_Y):
                    elt = Float32(sIn[i, tid_colwise_X])
                    sOut[i, tid_colwise_X] = (block_scale * elt).to(self.cfg.OUT_DTYPE)

            # Make the smem writes visible to the TMA engine before the store.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            if warp_idx == 0:
                cute.copy(
                    tma_atom_out,
                    tXs_out[(None, buff)],
                    tXg_out[(None, (tile_coord_y0 + it, tile_coord_x))],
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(BUFFERS_NUM - 1, read=True)
            # The bulk async-group is owned by the issuing thread. Hand its completion off to all
            # writers before the output ring can be reused.
            if cutlass.const_expr(BUFFERS_NUM <= next_it < ITERATIONS):
                cute.arch.sync_threads()

        if warp_idx == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=False)
        cute.arch.sync_threads()


def compile_dequantize_function_from_cfg(cfg: MXFP8DequantizeConfig):
    """Compile the dequantize kernel for cfg with symbolic shapes and a TVM-FFI entrypoint."""
    sym_M = cute.sym_int32()
    sym_N = cute.sym_int32(divisibility=SYM_N_DIVISIBILITY)
    # Scales are allocated at the padded shapes described in compile_cutedsl_function_from_cfg.
    if cfg.ROWWISE:
        scale_shape = (cute.sym_int32(divisibility=128), cute.sym_int32(divisibility=4))
    else:
        scale_shape = (cute.sym_int32(divisibility=4), cute.sym_int32(divisibility=128))

    def fake(dtype, shape, align):
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=align
        )

    return cute.compile(
        MXFP8DequantizeKernel(cfg),
        fake(cfg.FP8_DTYPE, (sym_M, sym_N), 16),  # mX
        fake(Float8E8M0FNU, scale_shape, 4),  # mS
        fake(cfg.OUT_DTYPE, (sym_M, sym_N), 16),  # mO
        cute.runtime.make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def get_mxfp8_dequantization_function(
    fn_name: str,
    fp8_dtype: str,
    out_dtype: str,
    rowwise: bool,
    with_gemm_swizzled_scales: bool,
) -> bool:
    """Compile the MXFP8 dequantize kernel for this config and register it in the TVM-FFI global
    registry under `fn_name`. Returns False if the config is unsupported, so the C++ dispatcher
    falls back to the CUDA kernel. Mirrors get_mxfp8_quantization_function.
    """
    try:
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
            cfg = MXFP8DequantizeConfig(fp8_dtype, out_dtype, rowwise, with_gemm_swizzled_scales)
        except ValueError as e:
            logger.warning(
                "CuTeDSL MXFP8 backend does not support this dequantize config, "
                "falling back to the CUDA C++ kernel: %s",
                e,
            )
            return False

        logger.debug("Compiling CuTeDSL MXFP8 dequantization kernel for %s", cfg)
        compiled = compile_dequantize_function_from_cfg(cfg)
        # Register the native function to skip the Python argument-parsing wrapper
        # (see get_mxfp8_quantization_function).
        native = getattr(compiled, "__tvm_ffi_object__", lambda: None)()
        tvm_ffi.register_global_func(
            fn_name, native if native is not None else compiled, override=True
        )
        return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "CuTeDSL MXFP8 dequantize kernel compilation & registration failed, falling back to"
            " the CUDA C++ kernel: %s",
            e,
        )
        return False
