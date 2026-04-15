# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 quantization kernel implemented in CuTeDSL.

Replicates the core logic of quantize_mxfp8.cuh: given a 2D tensor of BF16/FP16
values, quantize to MXFP8 format (FP8E4M3 data + E8M0 per-block scales).

The MXFP8 format uses 32-element blocks:
  - Rowwise:  each block is 32 contiguous elements along the column dimension.
              Scale shape: (M, N // 32)
  - Colwise:  each block is 32 contiguous elements along the row dimension.
              Scale shape: (M // 32, N)

Algorithm per block:
  1. Compute amax = max(|x_i|) for i in the 32-element block.
  2. Compute biased_exponent = float_to_e8m0(amax / max_fp8_value).
     This is the E8M0 scale stored as a uint8.
  3. Compute inverse_scale = 1 / 2^(biased_exponent - 127).
  4. For each element: out = (FP8)(x_i * inverse_scale).
"""

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int16, Uint8
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MXFP8_BLOCK_SIZE = 32  # Elements per MXFP8 scaling block

# FP8E4M3 max representable value = 448.0
FP8E4M3_MAX_NORM = 448.0
FP8E4M3_MAX_NORM_RCP = 1.0 / FP8E4M3_MAX_NORM

# FP8E5M2 max representable value = 57344.0
FP8E5M2_MAX_NORM = 57344.0
FP8E5M2_MAX_NORM_RCP = 1.0 / FP8E5M2_MAX_NORM

# FP32 constants
FP32_MANTISSA_BITS = 23


# ---------------------------------------------------------------------------
# Low-level DSL operations: bitcast, abs, E8M0 conversion
# ---------------------------------------------------------------------------
@dsl_user_op
def _bitcast_f32_to_i32(val: Float32, *, loc=None, ip=None) -> Int32:
    """Reinterpret a float32 as int32 (bitcast)."""
    return Int32(mlir_arith.bitcast(T.i32(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _bitcast_i32_to_f32(val: Int32, *, loc=None, ip=None) -> Float32:
    """Reinterpret an int32 as float32 (bitcast)."""
    return Float32(mlir_arith.bitcast(T.f32(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def fabs_f32(val: Float32, *, loc=None, ip=None) -> Float32:
    """Compute |val| by clearing the sign bit."""
    val_i32 = _bitcast_f32_to_i32(val, loc=loc, ip=ip)
    abs_i32 = val_i32 & Int32(0x7FFFFFFF)
    return _bitcast_i32_to_f32(abs_i32, loc=loc, ip=ip)


@dsl_user_op
def float_to_e8m0(val: Float32, *, loc=None, ip=None) -> Int32:
    """Convert a non-negative float to its E8M0 biased exponent (uint8 in int32).

    Branchless approach: adding 0x7FFFFF (all-ones mantissa) to the integer
    representation causes the mantissa to overflow into the exponent field
    exactly when mantissa > 0 (value is not an exact power of 2), giving
    the ceiling-exponent behavior required by the MXFP8 spec.
    Clamp to 254 for satfinite semantics.
    """
    val_i32 = _bitcast_f32_to_i32(val, loc=loc, ip=ip)
    rounded = val_i32 + Int32(0x7FFFFF)
    exponent = (rounded >> Int32(FP32_MANTISSA_BITS)) & Int32(0xFF)
    # min(exponent, 254) for satfinite
    clamped = Int32(
        mlir_arith.minsi(
            exponent.ir_value(loc=loc, ip=ip),
            Int32(254).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )
    return clamped


@dsl_user_op
def exp2f_rcp(biased_exp: Int32, *, loc=None, ip=None) -> Float32:
    """Compute 2^(127 - biased_exp) = 1 / 2^(biased_exp - 127).

    For the normal range [1, 253], this is __int_as_float((254 - biased_exp) << 23).
    Special cases handled via arith.select.
    """
    # Normal case
    new_exp = (Int32(254) - biased_exp) << Int32(FP32_MANTISSA_BITS)
    result = _bitcast_i32_to_f32(new_exp, loc=loc, ip=ip)

    # biased_exp == 255 → NaN
    is_255 = mlir_arith.cmpi(
        mlir_arith.CmpIPredicate.eq,
        biased_exp.ir_value(loc=loc, ip=ip),
        Int32(255).ir_value(loc=loc, ip=ip),
        loc=loc, ip=ip,
    )
    nan_val = _bitcast_i32_to_f32(Int32(0x7FFFFFFF), loc=loc, ip=ip)
    result = Float32(
        mlir_arith.select(
            is_255,
            nan_val.ir_value(loc=loc, ip=ip),
            result.ir_value(loc=loc, ip=ip),
            loc=loc, ip=ip,
        )
    )

    # biased_exp == 254 → 2^-127
    is_254 = mlir_arith.cmpi(
        mlir_arith.CmpIPredicate.eq,
        biased_exp.ir_value(loc=loc, ip=ip),
        Int32(254).ir_value(loc=loc, ip=ip),
        loc=loc, ip=ip,
    )
    small_val = _bitcast_i32_to_f32(Int32(0x00400000), loc=loc, ip=ip)
    result = Float32(
        mlir_arith.select(
            is_254,
            small_val.ir_value(loc=loc, ip=ip),
            result.ir_value(loc=loc, ip=ip),
            loc=loc, ip=ip,
        )
    )

    # biased_exp == 0 → 2^127 (input was zero, outputs will all be zero anyway)
    is_0 = mlir_arith.cmpi(
        mlir_arith.CmpIPredicate.eq,
        biased_exp.ir_value(loc=loc, ip=ip),
        Int32(0).ir_value(loc=loc, ip=ip),
        loc=loc, ip=ip,
    )
    huge_val = _bitcast_i32_to_f32(Int32(0x7F000000), loc=loc, ip=ip)
    result = Float32(
        mlir_arith.select(
            is_0,
            huge_val.ir_value(loc=loc, ip=ip),
            result.ir_value(loc=loc, ip=ip),
            loc=loc, ip=ip,
        )
    )

    return result


# ---------------------------------------------------------------------------
# FP32 → FP8E4M3 conversion via PTX
# ---------------------------------------------------------------------------
@dsl_user_op
def cvt_f32_to_fp8e4m3(val: Float32, *, loc=None, ip=None) -> Int32:
    """Convert float32 to fp8e4m3fn using PTX cvt instruction.

    Uses the hardware instruction ``cvt.rn.satfinite.e4m3x2.f32`` which
    converts two f32 values to two packed e4m3 values in a u16.
    We pass (0.0, val) so the low byte of the result contains the converted value.
    """
    zero = Float32(0.0)
    result_i16 = Int16(
        llvm.inline_asm(
            T.i16(),
            [zero.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
            "=h,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    # The low byte of the u16 contains convert(val) (second operand)
    result_i32 = Int32(
        mlir_arith.extui(T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )
    return result_i32 & Int32(0xFF)


# ---------------------------------------------------------------------------
# Kernel configuration
# ---------------------------------------------------------------------------
class MXFP8QuantizeConfig:
    """Configuration for the MXFP8 quantization kernel."""

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        M: int,
        N: int,
        fp8_dtype: str = "e4m3",
    ):
        self.dtype = dtype
        self.M = M
        self.N = N
        self.fp8_dtype = fp8_dtype

        if fp8_dtype == "e4m3":
            self.max_norm_rcp = FP8E4M3_MAX_NORM_RCP
        else:
            self.max_norm_rcp = FP8E5M2_MAX_NORM_RCP

        # Each thread handles one 32-element MXFP8 block.
        self.threads_per_row = self._compute_threads_per_row(N)
        self.num_threads = 128
        self.rows_per_block = self.num_threads // self.threads_per_row

    @staticmethod
    def _compute_threads_per_row(N: int) -> int:
        num_scale_cols = N // MXFP8_BLOCK_SIZE
        if num_scale_cols <= 4:
            return max(num_scale_cols, 1)
        elif num_scale_cols <= 8:
            return 8
        elif num_scale_cols <= 16:
            return 16
        elif num_scale_cols <= 32:
            return 32
        elif num_scale_cols <= 64:
            return 64
        else:
            return 128


# ---------------------------------------------------------------------------
# Rowwise MXFP8 quantization kernel
# ---------------------------------------------------------------------------
class MXFP8RowwiseQuantizeKernel:
    """Quantize a 2D tensor to MXFP8 along rows.

    Grid: (ceil(M / rows_per_block), ceil(num_scale_cols / threads_per_row))
    Block: (num_threads,)
    Each thread handles one 32-element block in one row.
    """

    def __init__(self, cfg: MXFP8QuantizeConfig):
        self.cfg = cfg

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        M: Int32,
        max_norm_rcp: Float32,
        stream: cuda.CUstream,
    ):
        cfg = self.cfg
        num_scale_cols = cfg.N // MXFP8_BLOCK_SIZE

        mX = cute.make_tensor(
            x_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)),
        )
        mO = cute.make_tensor(
            out_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)),
        )
        mS = cute.make_tensor(
            scale_ptr, cute.make_layout((M, num_scale_cols), stride=(num_scale_cols, 1)),
        )

        self.kernel(mX, mO, mS, max_norm_rcp).launch(
            grid=[
                cute.ceil_div(M, cfg.rows_per_block),
                cute.ceil_div(num_scale_cols, cfg.threads_per_row),
                1,
            ],
            block=[cfg.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        mS: cute.Tensor,
        max_norm_rcp: Float32,
    ):
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        M = mX.shape[0]
        num_scale_cols = cfg.N // MXFP8_BLOCK_SIZE

        row_in_block = tidx // cfg.threads_per_row
        col_in_block = tidx % cfg.threads_per_row
        global_row = bidx * cfg.rows_per_block + row_in_block
        scale_col = bidy * cfg.threads_per_row + col_in_block
        col_start = scale_col * MXFP8_BLOCK_SIZE

        row_valid = global_row < M
        col_valid = scale_col < num_scale_cols

        if row_valid & col_valid:
            # Step 1: Find amax over the 32-element block
            thread_amax = Float32(0.0)
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
                val = Float32(mX[global_row, col_start + i])
                abs_val = fabs_f32(val)
                thread_amax = cute.arch.fmax(thread_amax, abs_val)

            # Step 2: Compute E8M0 biased exponent (scale)
            biased_exponent = float_to_e8m0(thread_amax * max_norm_rcp)
            mS[global_row, scale_col] = Uint8(biased_exponent)

            # Step 3: Scale each element and convert to FP8
            inv_scale = exp2f_rcp(biased_exponent)
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
                val = Float32(mX[global_row, col_start + i])
                scaled = val * inv_scale
                fp8_byte = cvt_f32_to_fp8e4m3(scaled)
                mO[global_row, col_start + i] = Uint8(fp8_byte)


# ---------------------------------------------------------------------------
# Colwise MXFP8 quantization kernel
# ---------------------------------------------------------------------------
class MXFP8ColwiseQuantizeKernel:
    """Quantize a 2D tensor to MXFP8 along columns.

    Grid: (ceil(N / num_threads), M // 32)
    Block: (num_threads,)
    Each thread handles one 32-element block in one column.
    """

    def __init__(self, cfg: MXFP8QuantizeConfig):
        self.cfg = cfg

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        M: Int32,
        max_norm_rcp: Float32,
        stream: cuda.CUstream,
    ):
        cfg = self.cfg
        num_scale_rows = cfg.M // MXFP8_BLOCK_SIZE

        mX = cute.make_tensor(
            x_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)),
        )
        mO = cute.make_tensor(
            out_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)),
        )
        mS = cute.make_tensor(
            scale_ptr, cute.make_layout((num_scale_rows, cfg.N), stride=(cfg.N, 1)),
        )

        self.kernel(mX, mO, mS, max_norm_rcp).launch(
            grid=[cute.ceil_div(Int32(cfg.N), cfg.num_threads), num_scale_rows, 1],
            block=[cfg.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        mS: cute.Tensor,
        max_norm_rcp: Float32,
    ):
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        N = cfg.N
        global_col = bidx * cfg.num_threads + tidx
        scale_row_idx = bidy
        row_start = scale_row_idx * MXFP8_BLOCK_SIZE

        col_valid = global_col < Int32(N)

        if col_valid:
            thread_amax = Float32(0.0)
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
                val = Float32(mX[row_start + i, global_col])
                abs_val = fabs_f32(val)
                thread_amax = cute.arch.fmax(thread_amax, abs_val)

            biased_exponent = float_to_e8m0(thread_amax * max_norm_rcp)
            mS[scale_row_idx, global_col] = Uint8(biased_exponent)

            inv_scale = exp2f_rcp(biased_exponent)
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
                val = Float32(mX[row_start + i, global_col])
                scaled = val * inv_scale
                fp8_byte = cvt_f32_to_fp8e4m3(scaled)
                mO[row_start + i, global_col] = Uint8(fp8_byte)


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------
_compile_cache: dict = {}


def _get_compiled_kernel(kernel_cls, cfg, stream, direction):
    """Get or compile a kernel for the given configuration."""
    key = (direction, cfg.dtype, cfg.M, cfg.N, cfg.fp8_dtype)
    if key not in _compile_cache:
        kernel_obj = kernel_cls(cfg)
        compiled = cute.compile(
            kernel_obj,
            make_ptr(cfg.dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
            Int32(1),
            Float32(cfg.max_norm_rcp),
            stream,
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
    """Quantize a 2D tensor to MXFP8 format using a CuTeDSL kernel.

    Args:
        x: Input tensor of shape (M, N). Must be contiguous and on CUDA.
           dtype can be float16, bfloat16, or float32.
        fp8_dtype: Target FP8 format, "e4m3" or "e5m2".
        rowwise: If True, produce rowwise-quantized output.
        colwise: If True, produce colwise-quantized output.

    Returns:
        dict with keys (depending on rowwise/colwise flags):
          - "rowwise_data": (M, N) uint8 tensor (FP8 data)
          - "rowwise_scale": (M, N//32) uint8 tensor (E8M0 scales)
          - "colwise_data": (M, N) uint8 tensor (FP8 data)
          - "colwise_scale": (M//32, N) uint8 tensor (E8M0 scales)
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.ndim == 2, "Input must be 2D"

    M, N = x.shape
    assert rowwise or colwise, "At least one of rowwise/colwise must be True"
    if rowwise:
        assert N % MXFP8_BLOCK_SIZE == 0, f"N={N} must be divisible by {MXFP8_BLOCK_SIZE}"
    if colwise:
        assert M % MXFP8_BLOCK_SIZE == 0, f"M={M} must be divisible by {MXFP8_BLOCK_SIZE}"

    cutlass_dtype = _torch_to_cutlass_dtype[x.dtype]

    if fp8_dtype == "e4m3":
        max_norm_rcp = FP8E4M3_MAX_NORM_RCP
    else:
        max_norm_rcp = FP8E5M2_MAX_NORM_RCP

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    result = {}

    if rowwise:
        cfg = MXFP8QuantizeConfig(cutlass_dtype, M, N, fp8_dtype=fp8_dtype)
        compiled = _get_compiled_kernel(MXFP8RowwiseQuantizeKernel, cfg, stream, "rowwise")

        out_data = torch.empty((M, N), dtype=torch.uint8, device=x.device)
        out_scale = torch.empty((M, N // MXFP8_BLOCK_SIZE), dtype=torch.uint8, device=x.device)

        x_ptr = make_ptr(cutlass_dtype, x.data_ptr())
        out_ptr = make_ptr(Uint8, out_data.data_ptr())
        scale_ptr = make_ptr(Uint8, out_scale.data_ptr())

        compiled(x_ptr, out_ptr, scale_ptr, Int32(M), Float32(max_norm_rcp), stream)

        result["rowwise_data"] = out_data
        result["rowwise_scale"] = out_scale

    if colwise:
        cfg = MXFP8QuantizeConfig(cutlass_dtype, M, N, fp8_dtype=fp8_dtype)
        compiled = _get_compiled_kernel(MXFP8ColwiseQuantizeKernel, cfg, stream, "colwise")

        out_data = torch.empty((M, N), dtype=torch.uint8, device=x.device)
        out_scale = torch.empty((M // MXFP8_BLOCK_SIZE, N), dtype=torch.uint8, device=x.device)

        x_ptr = make_ptr(cutlass_dtype, x.data_ptr())
        out_ptr = make_ptr(Uint8, out_data.data_ptr())
        scale_ptr = make_ptr(Uint8, out_scale.data_ptr())

        compiled(x_ptr, out_ptr, scale_ptr, Int32(M), Float32(max_norm_rcp), stream)

        result["colwise_data"] = out_data
        result["colwise_scale"] = out_scale

    return result
