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

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int16, Uint8
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

# ---------------------------------------------------------------------------
# Constants — match quantize_mxfp8.cuh for the cast-only path
# ---------------------------------------------------------------------------
MXFP8_BLOCK_SIZE = 32   # alias used by tests
SCALE_DIM = 32           # Elements per MXFP8 scaling block (both dims)
CHUNK_DIM_Y = 64         # Rows per block
CHUNK_DIM_X = 64         # Cols per block
THREADS_PER_CHUNK = 64   # Threads per block
BUFF_DIM_Y = 32          # Rows per smem buffer = THREADS_Y
BUFF_DIM_X = 64          # Cols per smem buffer = CHUNK_DIM_X
STAGES = CHUNK_DIM_Y // BUFF_DIM_Y  # 2
THREADS_X = CHUNK_DIM_X // SCALE_DIM  # 2 scale-blocks per row
THREADS_Y = THREADS_PER_CHUNK // THREADS_X  # 32 rows per buffer

# FP8E4M3 max representable value
FP8E4M3_MAX_NORM = 448.0
FP8E4M3_MAX_NORM_RCP = 1.0 / FP8E4M3_MAX_NORM
FP8E5M2_MAX_NORM = 57344.0
FP8E5M2_MAX_NORM_RCP = 1.0 / FP8E5M2_MAX_NORM

FP32_MANTISSA_BITS = 23


# ---------------------------------------------------------------------------
# Low-level DSL operations
# ---------------------------------------------------------------------------
@dsl_user_op
def _bitcast_f32_to_i32(val: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(mlir_arith.bitcast(T.i32(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _bitcast_i32_to_f32(val: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(mlir_arith.bitcast(T.f32(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def fabs_f32(val: Float32, *, loc=None, ip=None) -> Float32:
    val_i32 = _bitcast_f32_to_i32(val, loc=loc, ip=ip)
    abs_i32 = val_i32 & Int32(0x7FFFFFFF)
    return _bitcast_i32_to_f32(abs_i32, loc=loc, ip=ip)


@dsl_user_op
def float_to_e8m0(val: Float32, *, loc=None, ip=None) -> Int32:
    """Branchless float->E8M0: add mantissa mask to round up, clamp to 254."""
    val_i32 = _bitcast_f32_to_i32(val, loc=loc, ip=ip)
    rounded = val_i32 + Int32(0x7FFFFF)
    exponent = (rounded >> Int32(FP32_MANTISSA_BITS)) & Int32(0xFF)
    return Int32(mlir_arith.minsi(
        exponent.ir_value(loc=loc, ip=ip),
        Int32(254).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def exp2f_rcp(biased_exp: Int32, *, loc=None, ip=None) -> Float32:
    """2^(127 - biased_exp) with special-case handling."""
    new_exp = (Int32(254) - biased_exp) << Int32(FP32_MANTISSA_BITS)
    result = _bitcast_i32_to_f32(new_exp, loc=loc, ip=ip)
    for (cmp_val, repl_bits) in [(255, 0x7FFFFFFF), (254, 0x00400000), (0, 0x7F000000)]:
        cond = mlir_arith.cmpi(mlir_arith.CmpIPredicate.eq,
                               biased_exp.ir_value(loc=loc, ip=ip),
                               Int32(cmp_val).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
        alt = _bitcast_i32_to_f32(Int32(repl_bits), loc=loc, ip=ip)
        result = Float32(mlir_arith.select(
            cond, alt.ir_value(loc=loc, ip=ip),
            result.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
    return result


@dsl_user_op
def cvt_f32_to_fp8e4m3(val: Float32, *, loc=None, ip=None) -> Int32:
    """float32 -> fp8e4m3fn via PTX cvt.rn.satfinite.e4m3x2.f32."""
    zero = Float32(0.0)
    result_i16 = Int16(llvm.inline_asm(
        T.i16(),
        [zero.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
        "=h,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
    result_i32 = Int32(mlir_arith.extui(
        T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
    return result_i32 & Int32(0xFF)


# ---------------------------------------------------------------------------
# Kernel configuration
# ---------------------------------------------------------------------------
class MXFP8QuantizeConfig:
    def __init__(self, dtype, M, N, fp8_dtype="e4m3", rowwise=True, colwise=False):
        self.dtype = dtype
        self.M = M
        self.N = N
        self.fp8_dtype = fp8_dtype
        self.rowwise = rowwise
        self.colwise = colwise
        self.max_norm_rcp = FP8E4M3_MAX_NORM_RCP if fp8_dtype == "e4m3" else FP8E5M2_MAX_NORM_RCP


# ---------------------------------------------------------------------------
# Rowwise MXFP8 quantization kernel — shared memory tiled
# ---------------------------------------------------------------------------
class MXFP8RowwiseSmemKernel:
    """Rowwise quantization with shared-memory tiling.

    Matches C++ kernel layout:
      Grid  (ceil(N/64), ceil(M/64))
      Block (64)
      Each block processes a 64x64 chunk in 2 stages of 32x64.

    Cooperative load: all 64 threads load one 32x64 tile (coalesced).
    Rowwise thread mapping (per stage):
      tid_Y = tidx // 2   -> row in [0, 32)
      tid_X = tidx %  2   -> scale-block in {0, 1}
      Each thread reads 32 contiguous elements from smem.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    @cute.jit
    def __call__(self, x_ptr, out_ptr, scale_ptr, M, max_norm_rcp, stream):
        cfg = self.cfg
        num_scale_cols = cfg.N // SCALE_DIM

        mX = cute.make_tensor(x_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)))
        mO = cute.make_tensor(out_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)))
        mS = cute.make_tensor(scale_ptr,
                              cute.make_layout((M, num_scale_cols), stride=(num_scale_cols, 1)))

        self.kernel(mX, mO, mS, max_norm_rcp).launch(
            grid=[cute.ceil_div(Int32(cfg.N), CHUNK_DIM_X),
                  cute.ceil_div(M, CHUNK_DIM_Y), 1],
            block=[THREADS_PER_CHUNK, 1, 1],
            smem=BUFF_DIM_Y * BUFF_DIM_X * (cfg.dtype.width // 8) + 128,
        )

    @cute.kernel
    def kernel(self, mX, mO, mS, max_norm_rcp):
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        M = mX.shape[0]
        N = cfg.N
        num_scale_cols = N // SCALE_DIM

        block_off_Y = bidy * CHUNK_DIM_Y
        block_off_X = bidx * CHUNK_DIM_X

        # Rowwise thread mapping
        tid_Y = tidx // THREADS_X
        tid_X = tidx % THREADS_X

        # Shared memory for one 32x64 input tile
        smem = utils.SmemAllocator()
        sX = smem.allocate_tensor(
            cfg.dtype,
            cute.make_ordered_layout((BUFF_DIM_Y, BUFF_DIM_X), order=(1, 0)),
            byte_alignment=128,
        )

        for stage in cutlass.range_constexpr(STAGES):
            base_row = block_off_Y + stage * BUFF_DIM_Y

            # Cooperative load: 64 threads x 32 rows (coalesced along cols)
            # M and N are required to be multiples of CHUNK_DIM at the API level
            for r in cutlass.range_constexpr(BUFF_DIM_Y):
                sX[r, tidx] = mX[base_row + r, block_off_X + tidx]
            cute.arch.barrier()

            # Rowwise computation — read 32 contiguous elements from smem
            g_row = base_row + tid_Y
            scale_col = bidx * THREADS_X + tid_X
            col_start = tid_X * SCALE_DIM

            amax = Float32(0.0)
            for i in cutlass.range_constexpr(SCALE_DIM):
                val = Float32(sX[tid_Y, col_start + i])
                amax = cute.arch.fmax(amax, fabs_f32(val))

            biased_exp = float_to_e8m0(amax * max_norm_rcp)
            mS[g_row, scale_col] = Uint8(biased_exp)

            inv_scale = exp2f_rcp(biased_exp)
            g_col_start = block_off_X + col_start
            for i in cutlass.range_constexpr(SCALE_DIM):
                val = Float32(sX[tid_Y, col_start + i])
                mO[g_row, g_col_start + i] = Uint8(cvt_f32_to_fp8e4m3(val * inv_scale))

            cute.arch.barrier()


# ---------------------------------------------------------------------------
# Colwise MXFP8 quantization kernel — shared memory tiled
# ---------------------------------------------------------------------------
class MXFP8ColwiseSmemKernel:
    """Colwise quantization with shared-memory tiling.

    Same grid/block as rowwise.  Per stage each of the 64 threads handles
    one column of the 32x64 tile (32 elements along rows read from smem).
    """

    def __init__(self, cfg):
        self.cfg = cfg

    @cute.jit
    def __call__(self, x_ptr, out_ptr, scale_ptr, M, max_norm_rcp, stream):
        cfg = self.cfg
        num_scale_rows = cfg.M // SCALE_DIM

        mX = cute.make_tensor(x_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)))
        mO = cute.make_tensor(out_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)))
        mS = cute.make_tensor(scale_ptr,
                              cute.make_layout((num_scale_rows, cfg.N), stride=(cfg.N, 1)))

        self.kernel(mX, mO, mS, max_norm_rcp).launch(
            grid=[cute.ceil_div(Int32(cfg.N), CHUNK_DIM_X),
                  cute.ceil_div(M, CHUNK_DIM_Y), 1],
            block=[THREADS_PER_CHUNK, 1, 1],
            smem=BUFF_DIM_Y * BUFF_DIM_X * (cfg.dtype.width // 8) + 128,
        )

    @cute.kernel
    def kernel(self, mX, mO, mS, max_norm_rcp):
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        M = mX.shape[0]
        N = cfg.N
        num_scale_rows = cfg.M // SCALE_DIM

        block_off_Y = bidy * CHUNK_DIM_Y
        block_off_X = bidx * CHUNK_DIM_X

        g_col = block_off_X + tidx

        smem = utils.SmemAllocator()
        sX = smem.allocate_tensor(
            cfg.dtype,
            cute.make_ordered_layout((BUFF_DIM_Y, BUFF_DIM_X), order=(1, 0)),
            byte_alignment=128,
        )

        for stage in cutlass.range_constexpr(STAGES):
            base_row = block_off_Y + stage * BUFF_DIM_Y

            # Cooperative load (M, N multiples of CHUNK_DIM — checked at API)
            for r in cutlass.range_constexpr(BUFF_DIM_Y):
                sX[r, tidx] = mX[base_row + r, block_off_X + tidx]
            cute.arch.barrier()

            # Colwise: thread tidx handles column tidx, 32 rows from smem
            scale_row = (block_off_Y + stage * BUFF_DIM_Y) // SCALE_DIM

            amax = Float32(0.0)
            for i in cutlass.range_constexpr(SCALE_DIM):
                val = Float32(sX[i, tidx])
                amax = cute.arch.fmax(amax, fabs_f32(val))

            biased_exp = float_to_e8m0(amax * max_norm_rcp)
            mS[scale_row, g_col] = Uint8(biased_exp)

            inv_scale = exp2f_rcp(biased_exp)
            for i in cutlass.range_constexpr(SCALE_DIM):
                val = Float32(sX[i, tidx])
                mO[base_row + i, g_col] = Uint8(cvt_f32_to_fp8e4m3(val * inv_scale))

            cute.arch.barrier()


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------
_compile_cache: dict = {}


def _get_compiled_kernel(kernel_cls, cfg, stream, direction):
    key = (direction, cfg.dtype, cfg.M, cfg.N, cfg.fp8_dtype)
    if key not in _compile_cache:
        kernel_obj = kernel_cls(cfg)
        compiled = cute.compile(
            kernel_obj,
            make_ptr(cfg.dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
            Int32(1), Float32(cfg.max_norm_rcp), stream,
        )
        _compile_cache[key] = compiled
    return _compile_cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_torch_to_cutlass_dtype = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def quantize_mxfp8_cutedsl(
    x: torch.Tensor,
    fp8_dtype: str = "e4m3",
    rowwise: bool = True,
    colwise: bool = False,
) -> dict:
    """Quantize a 2D tensor to MXFP8 format using CuTeDSL kernels with smem tiling."""
    assert x.is_cuda and x.is_contiguous() and x.ndim == 2
    M, N = x.shape
    assert rowwise or colwise
    assert M % CHUNK_DIM_Y == 0, f"M={M} must be a multiple of {CHUNK_DIM_Y}"
    assert N % CHUNK_DIM_X == 0, f"N={N} must be a multiple of {CHUNK_DIM_X}"

    cutlass_dtype = _torch_to_cutlass_dtype[x.dtype]
    max_norm_rcp = FP8E4M3_MAX_NORM_RCP if fp8_dtype == "e4m3" else FP8E5M2_MAX_NORM_RCP
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    result = {}

    if rowwise:
        cfg = MXFP8QuantizeConfig(cutlass_dtype, M, N, fp8_dtype, rowwise=True)
        compiled = _get_compiled_kernel(MXFP8RowwiseSmemKernel, cfg, stream, "rowwise")
        out_data = torch.empty((M, N), dtype=torch.uint8, device=x.device)
        out_scale = torch.empty((M, N // SCALE_DIM), dtype=torch.uint8, device=x.device)
        compiled(make_ptr(cutlass_dtype, x.data_ptr()),
                 make_ptr(Uint8, out_data.data_ptr()),
                 make_ptr(Uint8, out_scale.data_ptr()),
                 Int32(M), Float32(max_norm_rcp), stream)
        result["rowwise_data"] = out_data
        result["rowwise_scale"] = out_scale

    if colwise:
        cfg = MXFP8QuantizeConfig(cutlass_dtype, M, N, fp8_dtype, colwise=True)
        compiled = _get_compiled_kernel(MXFP8ColwiseSmemKernel, cfg, stream, "colwise")
        out_data = torch.empty((M, N), dtype=torch.uint8, device=x.device)
        out_scale = torch.empty((M // SCALE_DIM, N), dtype=torch.uint8, device=x.device)
        compiled(make_ptr(cutlass_dtype, x.data_ptr()),
                 make_ptr(Uint8, out_data.data_ptr()),
                 make_ptr(Uint8, out_scale.data_ptr()),
                 Int32(M), Float32(max_norm_rcp), stream)
        result["colwise_data"] = out_data
        result["colwise_scale"] = out_scale

    return result
