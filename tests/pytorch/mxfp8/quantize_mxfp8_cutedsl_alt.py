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

from types import SimpleNamespace
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


@dsl_user_op
def cvt_f32_to_fp8e5m2(val: Float32, *, loc=None, ip=None) -> Int32:
    """float32 -> fp8e5m2 via PTX cvt.rn.satfinite.e5m2x2.f32."""
    zero = Float32(0.0)
    result_i16 = Int16(llvm.inline_asm(
        T.i16(),
        [zero.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $2;",
        "=h,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
    result_i32 = Int32(mlir_arith.extui(
        T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
    return result_i32 & Int32(0xFF)


@dsl_user_op
def fma_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """`fma.rn.f32 d, a, b, c;` — single-instruction fused multiply-add
    matching nvcc's FFMA. Used for explicit `partial += a * b` patterns
    where we need the same rounding as TE's compiler-fused FFMA."""
    return Float32(llvm.inline_asm(
        T.f32(),
        [a.ir_value(loc=loc, ip=ip),
         b.ir_value(loc=loc, ip=ip),
         c.ir_value(loc=loc, ip=ip)],
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def tanh_approx(val: Float32, *, loc=None, ip=None) -> Float32:
    """`tanh.approx.f32` — fast tanh approximation. Matches CUDA `__tanhf`."""
    return Float32(llvm.inline_asm(
        T.f32(),
        [val.ir_value(loc=loc, ip=ip)],
        "tanh.approx.f32 $0, $1;",
        "=f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def pack_f32x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int64:
    """Pack two f32 scalars into a single 64-bit register (`floatx2` layout).

    Low 32 bits = `lo`, high 32 bits = `hi`. Uses `mov.b64 %dst, {%lo, %hi};`
    which lowers to a single register move — no actual memory traffic.
    """
    return Int64(llvm.inline_asm(
        T.i64(),
        [lo.ir_value(loc=loc, ip=ip), hi.ir_value(loc=loc, ip=ip)],
        "mov.b64 $0, {$1, $2};",
        "=l,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


# ---------------------------------------------------------------------------
# 16-bit packed input PTX kit (bf16 / f16)
#
# bf16 and f16 share the same fast-path shape: packed-x2 amax via
# `max.xorsign.abs.<fmt>x2`, then per-lane widen-to-f32 + `mul.f32x2` +
# `cvt.rn.satfinite.<out>x2.f32`. Only the opcodes differ. Build one PTX kit
# per format at module load and let the kernel pick the right kit at JIT
# trace time via `cfg.DTYPE` — equivalent to a C++ template arg specialization
# on `IType`, with no runtime branch.
# ---------------------------------------------------------------------------
def _build_packed16_kit(in_fmt: str):
    """Build a kit of PTX wrappers for a 16-bit input format.

    `in_fmt` is the PTX format string ('bf16' or 'f16'). Returns a namespace
    with the per-format ops the rowwise/colwise inner loops need:

      abs_max_x2(Int32, Int32)  -> Int32   # `max.xorsign.abs.<fmt>x2`
      abs_max_scalar(Int16, Int16) -> Int16  # `max.xorsign.abs.<fmt>`
      bits_to_f32(Int16) -> Float32          # widen one 16-bit element
      x2_lo_to_f32(Int32) -> Float32         # extract+widen low half
      x2_hi_to_f32(Int32) -> Float32         # extract+widen high half
      mul_cvt_to_fp8x2(fp8_dtype) -> callable(Int32, Int64)->Int32
                                            # fused <fmt>x2 * f32x2 -> fp8x2
    """

    @dsl_user_op
    def abs_max_x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
        return Int32(llvm.inline_asm(
            T.i32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            f"max.xorsign.abs.{in_fmt}x2 $0, $1, $2;",
            "=r,r,r", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))
    
    @dsl_user_op
    def max_x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
        return Int32(llvm.inline_asm(
            T.i32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            f"max.{in_fmt}x2 $0, $1, $2;",
            "=r,r,r", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))

    @dsl_user_op
    def abs_max_scalar(a: Int16, b: Int16, *, loc=None, ip=None) -> Int16:
        return Int16(llvm.inline_asm(
            T.i16(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            f"max.xorsign.abs.{in_fmt} $0, $1, $2;",
            "=h,h,h", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))

    if in_fmt == "bf16":
        # bf16 == top 16 bits of f32 — widening is a free bit-shift.
        @dsl_user_op
        def bits_to_f32(bits: Int16, *, loc=None, ip=None) -> Float32:
            i32 = Int32(mlir_arith.extui(
                T.i32(), bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
            return _bitcast_i32_to_f32(i32 << Int32(16), loc=loc, ip=ip)

        @dsl_user_op
        def x2_lo_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
            return _bitcast_i32_to_f32(
                (bits & Int32(0xFFFF)) << Int32(16), loc=loc, ip=ip)

        @dsl_user_op
        def x2_hi_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
            # `(x >> 16) << 16` ≡ `x & 0xFFFF0000`, sidestepping signed-literal
            # issues. Sign bits from the arith-right shift get zeroed by the
            # left shift.
            return _bitcast_i32_to_f32(
                (bits >> Int32(16)) << Int32(16), loc=loc, ip=ip)

        @dsl_user_op
        def truncate_f32(val: Float32, *, loc=None, ip=None) -> Float32:
            """Round f32 to bf16 precision (round-to-nearest-even), keep f32.
            Matches C++'s `static_cast<float>(static_cast<bf16>(elt))`."""
            bf16_bits = Int16(llvm.inline_asm(
                T.i16(), [val.ir_value(loc=loc, ip=ip)],
                "cvt.rn.bf16.f32 $0, $1;",
                "=h,f", has_side_effects=False, is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT))
            i32 = Int32(mlir_arith.extui(
                T.i32(), bf16_bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
            return _bitcast_i32_to_f32(i32 << Int32(16), loc=loc, ip=ip)
    else:
        # f16 has its own bit layout; widening requires `cvt.f32.f16`.
        @dsl_user_op
        def bits_to_f32(bits: Int16, *, loc=None, ip=None) -> Float32:
            return Float32(llvm.inline_asm(
                T.f32(), [bits.ir_value(loc=loc, ip=ip)],
                "cvt.f32.f16 $0, $1;",
                "=f,h", has_side_effects=False, is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT))

        @dsl_user_op
        def x2_lo_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
            lo_i16 = Int16(mlir_arith.trunci(
                T.i16(), bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
            return bits_to_f32(lo_i16, loc=loc, ip=ip)

        @dsl_user_op
        def x2_hi_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
            hi_shifted = bits >> Int32(16)
            hi_i16 = Int16(mlir_arith.trunci(
                T.i16(), hi_shifted.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
            return bits_to_f32(hi_i16, loc=loc, ip=ip)

        @dsl_user_op
        def truncate_f32(val: Float32, *, loc=None, ip=None) -> Float32:
            """Round f32 to f16 precision, keep f32."""
            f16_bits = Int16(llvm.inline_asm(
                T.i16(), [val.ir_value(loc=loc, ip=ip)],
                "cvt.rn.f16.f32 $0, $1;",
                "=h,f", has_side_effects=False, is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT))
            return Float32(llvm.inline_asm(
                T.f32(), [f16_bits.ir_value(loc=loc, ip=ip)],
                "cvt.f32.f16 $0, $1;",
                "=f,h", has_side_effects=False, is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT))

    def _build_mul_cvt(out_fmt: str, relu: bool = False):
        """Build a fused `<in_fmt>x2 * f32x2 → fp8<out_fmt>x2` PTX wrapper.

        The shape is identical across (in_fmt, out_fmt) combos — only the
        widening opcode (`cvt.f32.<in_fmt>`) and the final saturating cvt
        (`cvt.rn.satfinite.<out_fmt>x2.f32`) differ.
        """
        out_op = "e4m3x2" if out_fmt == "e4m3" else "e5m2x2"
        asm = (
            "{\n"
            ".reg.b64 vp0; .reg.b64 vp1;\n\t"
            ".reg.b32 v1;  .reg.b32 v2;\n\t"
            ".reg.b16 vb1; .reg.b16 vb2;\n\t"
            "mov.b32 {vb1, vb2}, $1;\n\t"
            f"cvt.f32.{in_fmt} v1, vb1;\n\t"
            f"cvt.f32.{in_fmt} v2, vb2;\n\t"
            "mov.b64 vp0, {v1, v2};\n\t"
            "mul.f32x2 vp1, vp0, $2;\n\t"
            "mov.b64 {v2, v1}, vp1;\n\t"
            f"cvt.rn.satfinite{".relu" if relu else ""}.{out_op}.f32 $0, v1, v2;\n\t"
            "}"
        )

        @dsl_user_op
        def fn(val_2x: Int32, scale_2x: Int64, *, loc=None, ip=None) -> Int32:
            result_i16 = Int16(llvm.inline_asm(
                T.i16(),
                [val_2x.ir_value(loc=loc, ip=ip),
                 scale_2x.ir_value(loc=loc, ip=ip)],
                asm,
                "=h,r,l", has_side_effects=False, is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT))
            return Int32(mlir_arith.extui(
                T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
        return fn

    def mul_cvt_to_fp8x2(fp8_dtype: str, relu: bool = False):
        if fp8_dtype == "e5m2":
            return _build_mul_cvt("e5m2", relu)
        return _build_mul_cvt("e4m3", relu)

    return SimpleNamespace(
        abs_max_x2=abs_max_x2,
        max_x2=max_x2,
        abs_max_scalar=abs_max_scalar,
        bits_to_f32=bits_to_f32,
        x2_lo_to_f32=x2_lo_to_f32,
        x2_hi_to_f32=x2_hi_to_f32,
        truncate_f32=truncate_f32,
        mul_cvt_to_fp8x2=mul_cvt_to_fp8x2,
    )


_BF16_KIT = _build_packed16_kit("bf16")
_F16_KIT = _build_packed16_kit("f16")


def _is_packed16(dtype) -> bool:
    """True if `dtype` is one of the 16-bit packed input formats."""
    return dtype is cutlass.BFloat16 or dtype is cutlass.Float16


def _packed16_kit(dtype):
    """Trace-time selector — pick a Packed16Kit for the input dtype."""
    if dtype is cutlass.Float16:
        return _F16_KIT
    return _BF16_KIT


# ---------------------------------------------------------------------------
# Forward-activation registry
#
# Each entry is a Float32 → Float32 callable applied per element before the
# MXFP8 amax + cast. Selection is by Python string at JIT trace time, so the
# const-expr machinery treats `cfg.ACTIVATION` like a C++ template argument
# — no runtime branch in the inner loop, separate kernel cached per choice.
#
# Math primitives match CUDA fast-math intrinsics so outputs are bit-exact
# with PyTorch's CUDA implementations of the same activations:
#   tanh   -> tanh.approx.f32 (== __tanhf)
#   exp(x) -> exp2.approx.f32(x · log2(e)) (== __expf)
# ---------------------------------------------------------------------------
def _act_relu(x: Float32) -> Float32:
    return cute.arch.fmax(x, Float32(0.0))


def _act_gelu(x: Float32) -> Float32:
    """Tanh-approximation GELU. Constants and operator grouping match TE's
    `transformer_engine/common/util/math.h::gelu` exactly (factored form
    `x · (0.5 + 0.5·tanh(x·(a + b·x²)))`) so quantized output is bit-exact
    against the C++ fused IS_ACT path. Uses `cute.math.tanh(fastmath=False)`
    rather than the `tanh.approx.f32` PTX intrinsic — TE compiles activation
    kernels without `--use_fast_math` by default, so its `tanhf` is the
    IEEE-precise expansion."""
    A = Float32(0.79788456)       # sqrt(2/π) truncated to TE's 8-digit literal
    B = Float32(0.03567741)       # = sqrt(2/π) · 0.044715, same truncation
    return x * (Float32(0.5) + Float32(0.5) * cute.math.tanh(x * (A + B * x * x)))


def _act_silu(x: Float32) -> Float32:
    """SiLU/Swish: x · σ(x) = x / (1 + e^-x).
    Matches TE's `silu` (`val / (1 + expf(-val))`)."""
    return x / (Float32(1.0) + cute.arch.exp(-x))


# ---- Backward (derivative) activations ----
# Used by IS_DACT paths: kernel computes `elt = grad_y · dOP(act_in)`. The
# entries below take `act_in` and return the derivative of the activation
# evaluated at it; the kernel multiplies by `grad_y` afterwards. Constants
# match TE's `transformer_engine/common/util/math.h` exactly so quantized
# output is bit-equal to `tex.dgelu`/`tex.dsilu`/`tex.drelu`.
@dsl_user_op
def _act_drelu(x: Float32, *, loc=None, ip=None) -> Float32:
    """drelu(x) = x > 0 ? 1.0 : 0.0 (NaN → 0 per IEEE OGT). Matches TE's drelu."""
    pred = mlir_arith.cmpf(
        mlir_arith.CmpFPredicate.OGT,
        x.ir_value(loc=loc, ip=ip),
        Float32(0.0).ir_value(loc=loc, ip=ip),
        loc=loc, ip=ip,
    )
    return Float32(mlir_arith.select(
        pred,
        Float32(1.0).ir_value(loc=loc, ip=ip),
        Float32(0.0).ir_value(loc=loc, ip=ip),
        loc=loc, ip=ip,
    ))


def _act_dgelu(x: Float32) -> Float32:
    """tanh-approximation GELU derivative.
    Mirrors TE's dgelu: tanh(`A·x·(1 + κ·x²)`) — note the inner-arg shape
    differs slightly from forward gelu's `x·(A + B·x²)` form (κ vs B); this
    is TE's exact source, preserved for bit-exact match."""
    A = Float32(0.79788456)
    KAPPA = Float32(0.044715)
    C = Float32(0.1070322243)   # = 3·κ·A, full-precision constant TE uses
    inner = A * x * (Float32(1.0) + KAPPA * x * x)
    tanh_out = cute.math.tanh(inner)
    one_minus_tanh_sq = Float32(1.0) - tanh_out * tanh_out
    return (Float32(0.5) * x * (one_minus_tanh_sq * (A + C * x * x))
            + Float32(0.5) * (Float32(1.0) + tanh_out))


def _act_dsilu(x: Float32) -> Float32:
    """dsilu(x) = x · σ(x)·(1 - σ(x)) + σ(x). Matches TE's dsilu via
    `cval * dsigmoid(cval) + sigmoid(cval)` after inlining."""
    s = Float32(1.0) / (Float32(1.0) + cute.arch.exp(-x))
    return x * (s * (Float32(1.0) - s)) + s


_ACTIVATIONS = {
    "relu": _act_relu,
    "gelu": _act_gelu,
    "silu": _act_silu,
    "drelu": _act_drelu,
    "dgelu": _act_dgelu,
    "dsilu": _act_dsilu,
}


def _is_derivative_activation(name) -> bool:
    """True if `name` is one of the registered backward (derivative) activations."""
    return isinstance(name, str) and name.startswith("d") and name in _ACTIVATIONS


@dsl_user_op
def cvt_f32x2_to_fp8e4m3x2(val_hi: Float32, val_lo: Float32, relu: bool = False,
                             *, loc=None, ip=None) -> Int32:
    """Convert two float32 values to two packed fp8e4m3fn bytes in one instruction.

    Returns an int32 where bits [7:0] = fp8(val_lo), bits [15:8] = fp8(val_hi).
    This mirrors ptx::mul_cvt_2x which converts 2 values in one instruction.
    """
    result_i16 = Int16(llvm.inline_asm(
        T.i16(),
        [val_hi.ir_value(loc=loc, ip=ip), val_lo.ir_value(loc=loc, ip=ip)],
        f"cvt.rn.satfinite{".relu" if relu else ""}.e4m3x2.f32 $0, $1, $2;",
        "=h,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
    return Int32(mlir_arith.extui(
        T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def cvt_f32x2_to_fp8e5m2x2(val_hi: Float32, val_lo: Float32, relu: bool = False,
                             *, loc=None, ip=None) -> Int32:
    """e5m2 sibling of `cvt_f32x2_to_fp8e4m3x2`."""
    result_i16 = Int16(llvm.inline_asm(
        T.i16(),
        [val_hi.ir_value(loc=loc, ip=ip), val_lo.ir_value(loc=loc, ip=ip)],
        f"cvt.rn.satfinite{".relu" if relu else ""}.e5m2x2.f32 $0, $1, $2;",
        "=h,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
    return Int32(mlir_arith.extui(
        T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def _cvt_f32_to_fp8(fp8_dtype: str):
    """Const-expr dispatch: pick the f32→fp8 scalar PTX op based on output dtype.

    `fp8_dtype` is the Python string from `cfg.FP8_DTYPE`, evaluated at JIT
    trace time; the unused branch is never traced.
    """
    if fp8_dtype == "e5m2":
        return cvt_f32_to_fp8e5m2
    return cvt_f32_to_fp8e4m3


def _cvt_f32x2_to_fp8x2(fp8_dtype: str):
    """Const-expr dispatch for the packed f32x2→fp8x2 cvt."""
    if fp8_dtype == "e5m2":
        return cvt_f32x2_to_fp8e5m2x2
    return cvt_f32x2_to_fp8e4m3x2

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


class DoNothingKernel:
    @cute.jit
    def __call__(
        self,
        x0: cute.Tensor,
        x1: cute.Tensor,
        x2: cute.Tensor,
        x3: cute.Tensor,
        x4: cute.Tensor,
        x5: cute.Tensor,
        x6: cute.Tensor,
        x7: cute.Tensor,
        x8: cute.Tensor,
    ):
        n = 1

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
            num_tiles_M = (cfg.M + 127) // 128
            num_tiles_SC = (num_scale_cols + 3) // 4   # = ceil(N / 128)
            num_tiles_SR = (num_scale_rows + 3) // 4   # = ceil(M / 128)
            num_tiles_N = (cfg.N + 127) // 128
            # row i = i_lo + 32 * (i_hi + 4 * tile_Y);  col j = j_lo + 4 * tile_X.
            # Within one 128×4 tile: byte = i_lo*16 + i_hi*4 + j_lo.
        
            # Tile-major outer dims add (tile_Y * num_tiles_SC + tile_X) * 512.
            # For example, if M=256, N=512, then num_scale_cols = 16, num_scale_rows = 8, and num_tiles_M=2, num_tiles_SC=4, num_tiles_SR=2, num_tiles_N=4
            # The swizzled layout is ((32, 4, 2), (4, 4)):((16, 4, 2048), (1, 512))
            mS_row = cute.make_tensor(
                mS_row.iterator,
                cute.make_layout(
                    ((32, 4, num_tiles_M), (4, num_tiles_SC)),
                    stride=((16, 4, num_tiles_SC * 512), (1, 512)),
                ),
            )
            # Colwise: same swizzle, axes swap roles — col axis gets the 32×4
            # inner decomp, scale-row axis gets the 4-extent dim.
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
        
        if cutlass.const_expr(cfg.IS_DACT):
            # Second input descriptor — only consumed by the IS_DACT path. When
            # not is_dact, the wrapper aliases this onto x's data ptr so the
            # descriptor is well-formed but never read.
            tma_atom_act, tma_src_act = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_load, mActIn, smem_tile_layout, cta_tiler, num_multicast=1,
            )
        else:
            tma_atom_act = None
            tma_src_act = None
        
        # Output: TMA S2G (uint8 smem → gmem) for both directions. Creating
        # both atoms unconditionally — if a direction is disabled the kernel
        # simply won't dispatch its copy, and the atom cost is negligible.
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        out_smem_layout = cute.make_ordered_layout((TILE_Y, TILE_X), order=(1, 0))
        if cutlass.const_expr(cfg.ROWWISE):
            tma_atom_out_row, tma_dst_out_row = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store, mO_row, out_smem_layout, cta_tiler, num_multicast=1,
            )
            tma_atom_out_col = None
            tma_dst_out_col = None
        if cutlass.const_expr(cfg.COLWISE):
            tma_atom_out_col, tma_dst_out_col = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store, mO_col, out_smem_layout, cta_tiler, num_multicast=1,
            )
            tma_atom_out_row = None
            tma_dst_out_row = None
        
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
        mX, # (M, N):(N, 1), the tensor to quantize
        mO_row, # (M, N):(N, 1), rowwise quantized output tensor (uint8)
        mS_row, # TODO: What is this
        mO_col, # (M, N):(N, 1), colwise quantized output tensor (uint8)
        mS_col, # (M // 32, N):(N, 1), colwise scale tensor (uint8)
        mNoop,  # (1,) f32 — skip all work if mNoop[0] == 1.0
        mAmax,  # (1,) f32 — global amax accumulator (only used if cfg.WITH_AMAX)
        mDbias, # (blocks_Y, N) f32 dbias workspace (only used if cfg.WITH_DBIAS)
        max_norm_rcp,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom, tma_src, # how to use TMA to copy the input
        tma_atom_act, tma_src_act,   # only used by the IS_DACT path
        tma_atom_out_row, tma_dst_out_row, # how to use TMA to copy the rowwise output
        tma_atom_out_col, tma_dst_out_col, # how to use TMA to copy the colwise output
    ):
        cfg = self.cfg

        mS_row = cute.zipped_divide(mS_row, (TILE_Y, TILE_X // SCALE_DIM))
        # For M=256, N=512:
        # Non-swizzled: https://kainzhong.github.io/CuTe-Layout-Visualizer/?key=zipped_divide-%28256%2C+16%29%3A%2816%2C+1%29-32%0A2
        # Swizzled: https://kainzhong.github.io/CuTe-Layout-Visualizer/?key=zipped_divide-%28%2832%2C+4%2C+2%29%2C+%284%2C+4%29%29%3A%28%2816%2C+4%2C+2048%29%2C+%281%2C+512%29%29-32%0A2
        # print(f"mS_row after zipped_divide: {mS_row}")


        # ---- noop early-exit ----------------------------------------------
        # Skip the entire kernel if the framework signalled "no-op" via a
        # 1-element f32 flag in gmem. Used by CUDA Graphs / MoE to gate work
        # without a host-GPU sync. All threads in the CTA see the same value
        # → uniform branch, no divergence, no setup to unwind since this runs
        # before any smem alloc / mbarrier init.
        #
        # CuTeDSL forbids a runtime `return` inside @cute.kernel, so we wrap
        # the body in `_kernel_main*` and just guard the call here.
        # Bitcast f32 → i32 so we can compare against 1.0's bit pattern via
        # the well-tested int-compare path.
        noop_bits = _bitcast_f32_to_i32(mNoop[0])
        if noop_bits != Int32(0x3F800000):
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

            if cutlass.const_expr(cfg.COLWISE):
                # The first row that belongs to this CTA. Each CTA handles NUM_TILES of (TILE_Y, TILE_X) tiles stacked vertically,
                # and each stage handles one of them.
                base_row = (bidy * NUM_TILES + stage) * TILE_Y
                sO_col_tile = sO_col[(None, stage)]
                amax_c, block_dbias = self._process_colwise(
                    sX_tile, sO_col_tile, base_row, bidx, tidx,
                    mS_col, max_norm_rcp, block_dbias,
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
                row_tile_idx = bidy * NUM_TILES + stage
                mS_row_stage = cute.flatten(mS_row[(None, (row_tile_idx, bidx))])
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
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
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
    def _process_colwise(
        self,
        sX_tile,        # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
        sO_col_tile,    # (TILE_Y, TILE_X) uint8 smem view (colwise FP8 output)
        base_row,       # Int32: global Y offset of this tile's first row
        bidx,           # Int32: block X index (column tile)
        tidx,           # Int32: thread index within the CTA
        mS_col,         # colwise scale tensor (1D swizzled, or 2D linear)
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
        block_off_X = bidx * TILE_X
        col_global = block_off_X + tidx

        # Flat views (sX[(None, stage)] has nested shape ((TILE_Y, TILE_X),)).
        sX_flat = cute.make_tensor(
            sX_tile.iterator,
            cute.make_layout((TILE_Y, TILE_X), stride=(TILE_X, 1)),
        )
        sO_col_flat = cute.make_tensor(
            sO_col_tile.iterator,
            cute.make_layout((TILE_Y, TILE_X), stride=(TILE_X, 1)),
        )

        # 0. Load the 32-element column from smem into registers once (matches
        # C++'s `in_colwise_IType[i]` cache). Amax and cast both reuse these.
        # Path selection:
        #   - 16-bit input WITHOUT activation AND without dbias: packed-x2
        #     amax in IType, fast.
        #   - everything else (16-bit + activation, with_dbias, OR fp32 input):
        #     scalar f32 path. With activation, apply OP and (for 16-bit
        #     input) round-trip through IType to match C++'s
        #     `static_cast<IType>(elt)` numerical truncation. with_dbias
        #     accumulates the per-column sum BEFORE truncation (C++ order).
        partial_dbias = partial_dbias_in
        if cutlass.const_expr(_is_packed16(cfg.DTYPE) and cfg.ACTIVATION is None
                              and not cfg.WITH_DBIAS):
            kit = _packed16_kit(cfg.DTYPE)
            sX_i16 = cute.make_tensor(
                cute.recast_ptr(sX_tile.iterator, dtype=Int16),
                cute.make_layout((TILE_Y, TILE_X), stride=(TILE_X, 1)),
            )
            in_c = [sX_i16[i, tidx] for i in range(SCALE_DIM)]

            amax_bits = Int16(0)
            for i in cutlass.range_constexpr(SCALE_DIM):
                amax_bits = kit.abs_max_scalar(amax_bits, in_c[i])
            amax_c = fabs_f32(kit.bits_to_f32(amax_bits))
        else:
            in_c = [Float32(sX_flat[i, tidx]) for i in range(SCALE_DIM)]
            # Apply activation in f32 (no truncation yet — dbias must
            # accumulate from the pre-truncation value to match C++ order).
            if cutlass.const_expr(cfg.ACTIVATION is not None):
                op = _ACTIVATIONS[cfg.ACTIVATION]
                for i in cutlass.range_constexpr(SCALE_DIM):
                    in_c[i] = op(in_c[i])
            # Accumulate per-column dbias from f32 (pre-truncation) values.
            # IMPORTANT: caller passes the running block_dbias accumulator and
            # we extend it one element at a time. This matches C++'s flat
            # `partial_dbias_colwise += elt` order across the inner loop —
            # grouping by stage (stage0_sum + stage1_sum) rounds slightly
            # differently and produces ULP-level fp32 mismatches.
            if cutlass.const_expr(cfg.WITH_DBIAS):
                for i in cutlass.range_constexpr(SCALE_DIM):
                    partial_dbias = partial_dbias + in_c[i]
            # Numerical truncation through IType so amax/cast match C++.
            # Only needed when 16-bit input + activation; without activation
            # the widening was already exact.
            if cutlass.const_expr(_is_packed16(cfg.DTYPE)
                                  and cfg.ACTIVATION is not None):
                kit_act = _packed16_kit(cfg.DTYPE)
                for i in cutlass.range_constexpr(SCALE_DIM):
                    in_c[i] = kit_act.truncate_f32(in_c[i])
            amax_c = Float32(0.0)
            for i in cutlass.range_constexpr(SCALE_DIM):
                amax_c = cute.arch.fmax(amax_c, fabs_f32(in_c[i]))

        # 2. E8M0 scale → gmem. mS_col's layout already encodes the swizzle
        # when cfg.WITH_GEMM_SWIZZLED_SCALES=True, so 2D access just works.
        biased_exp_c = float_to_e8m0(amax_c * max_norm_rcp)
        scale_row = base_row // SCALE_DIM
        mS_col[scale_row, col_global] = Uint8(biased_exp_c)

        # 3. scale + FP8 cast → smem (one byte per (row, tidx)). Caller
        # flushes the whole (TILE_Y, TILE_X) tile with a TMA S2G.
        inv_scale_c = exp2f_rcp(biased_exp_c)
        cvt_to_fp8 = _cvt_f32_to_fp8(cfg.FP8_DTYPE)
        if cutlass.const_expr(_is_packed16(cfg.DTYPE) and cfg.ACTIVATION is None
                              and not cfg.WITH_DBIAS):
            kit_cast = _packed16_kit(cfg.DTYPE)
            for i in cutlass.range_constexpr(SCALE_DIM):
                v_f32 = kit_cast.bits_to_f32(in_c[i])
                sO_col_flat[i, tidx] = Uint8(cvt_to_fp8(v_f32 * inv_scale_c))
        else:
            # in_c[i] is already Float32 in the activation/f32 path.
            for i in cutlass.range_constexpr(SCALE_DIM):
                sO_col_flat[i, tidx] = Uint8(cvt_to_fp8(in_c[i] * inv_scale_c))

        # Per-thread amax + per-thread per-column dbias accumulator. Both
        # are folded across stages by the caller; dbias is later written to
        # the workspace and reduced externally.
        return amax_c, partial_dbias

    @cute.jit
    def _process_rowwise(
        self,
        sX_tile,        # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
        sO_row_tile,    # (TILE_Y, TILE_X) uint8 smem view (rowwise FP8 output)
        mS_row_stage,         # rowwise scale tensor (1D swizzled, or 2D linear)
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
        tidx, _, _ = cute.arch.thread_idx()

        # Match the C++ reference's thread layout: pairs of adjacent lanes
        # share a row (lanes 2k / 2k+1 both own row k), each pair covering
        # the two 32-element scale blocks of that row. Express as a cute
        # layout mapping `(tid_Y, tid_X) -> tidx` with stride (2, 1):
        # linear(tidx) = tid_Y*2 + tid_X, so `get_flat_coord` inverts to
        # `(tidx // 2, tidx % 2)` — semantically clearer than the raw
        # divmod, and readily reusable if we later partition via TiledCopy.
        # print(f"sX_tile: {sX_tile}")
        # print(f"sO_row_tile: {sO_row_tile}")
        # print(f"mS_row_stage: {mS_row_stage}")
        
        tiler, tv_layout = cute.make_layout_tv(
            thr_layout=cute.make_layout((TILE_Y, 2), stride=(2, 1)),
            val_layout=cute.make_layout((1, SCALE_DIM), stride=(0, 1))
        )
        # print(f"tv_layout: {tv_layout}")
        # print(f"tiler: {tiler}")
        
        sX_tv = cute.composition(sX_tile, tv_layout)
        sO_tv = cute.composition(sO_row_tile, tv_layout)

        # I/O Elements that belong to this thread
        sX_thread = sX_tv[tidx, None]   # shape (32,) bf16
        sO_thread = sO_tv[tidx, None]   # shape (32,) uint8

        # See https://kainzhong.github.io/CuTe-Layout-Visualizer/?key=tv-2-%2832%2C+2%29%3A%282%2C1%29-%281%2C+32%29%3A%280%2C1%29
        # print(f"sX_thread: {sX_thread}")
        # print(f"sO_thread: {sO_thread}")

        sO_thread_u32_ptr = cute.recast_ptr(sO_thread.iterator, dtype=Uint32)
        # Each wave it writes 32 bytes = 8 uint32s, so in 4 waves we write all 32 quantized elements.
        sO_thread_u32 = cute.make_tensor(
            sO_thread_u32_ptr,
            cute.make_layout((SCALE_DIM // 4,), stride=(1,)), # 1 uint32 is 4 fp8 elements
        )
        # print(f"sO_thread_u32: {sO_thread_u32}")

        thread_dbias = thread_dbias_in

        FUSE_RELU = cutlass.const_expr(cfg.ACTIVATION == "relu" and not cfg.WITH_DBIAS)
        # For this fast paht we can read in pack of 2 instead of reading individual f16 / bf16 element
        _row_fast = (_is_packed16(cfg.DTYPE) and (cfg.ACTIVATION is None or FUSE_RELU)
                     and not (cfg.WITH_DBIAS and not cfg.COLWISE))

        if cutlass.const_expr(_row_fast):
            # If no activation, no dbias, f16 / bf16 and rowwise quantization, we can read 2 f16 / bf16 at once in a pack
            # and use max.xorsign.abs.f16x2 / max.xorsign.abs.bf16x2 to compute
            kit = _packed16_kit(cfg.DTYPE)
            sX_thread_rw_i32 = cute.make_tensor(
                cute.recast_ptr(sX_thread.iterator, dtype=Int32),
                cute.make_layout((1, SCALE_DIM // 2), stride=(0, 1)), # 1 int32 is 2 fp16/bf16 elements
            )
            # print(f"sX_thread_rw_i32: {sX_thread_rw_i32}")
            # Each wave we read 2 packed i32, which is 4 fp16/bf16 elements (PACK_SIZE)
            # In total we have 8 waves where each wave reads 
            in_r = [[None, None] for _ in range(WAVES)]
            bank_group = (tidx % THREADS_PER_WARP) // THREADS_PER_BANK # Each 4 threads share the same bank, which forms a bank group
            offset = bank_group * 2 # Each bank group will read 2 i32 from their bank
            for w in cutlass.range_constexpr(WAVES):
                idx = (w * 2 + offset) % (SCALE_DIM // 2)
                in_r[w][0] = sX_thread_rw_i32[0, idx]
                in_r[w][1] = sX_thread_rw_i32[0, idx + 1]

            # 1. Packed-x2 amax — 2 PTX per wave, 16 total per thread.
            # Accumulates `|elt|` in both lanes (with xorsign-drifted signs);
            # final horizontal max reduces the two lanes to a single f32.
            amax_2x = Int32(0)
            # Each wave will use max.xorsign.abs.f16x2 or max.xorsign.abs.bf16x2 to compare 2 packed elements in parallel
            for w in cutlass.range_constexpr(WAVES):
                if cutlass.const_expr(FUSE_RELU):
                    # If we fuse relu then we don't want to do abs since negative value will be set to 0 and they will lose comparison automatically
                    amax_2x = kit.max_x2(amax_2x, in_r[w][0])
                    amax_2x = kit.max_x2(amax_2x, in_r[w][1])
                else:
                    amax_2x = kit.abs_max_x2(amax_2x, in_r[w][0])
                    amax_2x = kit.abs_max_x2(amax_2x, in_r[w][1])
            if cutlass.const_expr(FUSE_RELU):
                # Compare the 2 packed max without abs
                amax_r = cute.arch.fmax(
                    kit.x2_lo_to_f32(amax_2x),
                    kit.x2_hi_to_f32(amax_2x),
                )
                # For relu the max is at least 0
                if cutlass.const_expr(FUSE_RELU):
                    amax_r = cute.arch.fmax(amax_r, Float32(0.0))
            else:
                # Compare the 2 packed abs max
                amax_r = cute.arch.fmax(
                    fabs_f32(kit.x2_lo_to_f32(amax_2x)),
                    fabs_f32(kit.x2_hi_to_f32(amax_2x)),
                )
        else:
            # Since we need to do computation on individual f16 / bf16 elements, we can't read in pack
            sX_thread_rw = cute.make_tensor(
                sX_thread.iterator,
                cute.make_layout((1, SCALE_DIM), stride=(0, 1)),
            )
            in_r = [[None] * PACK_SIZE for _ in range(WAVES)]
            bank_group = (tidx % THREADS_PER_WARP) // THREADS_PER_BANK # Each 4 threads share the same bank, which forms a bank group
            offset = bank_group * 4 # Each bank group will read 4 f16 from their bank

            if cutlass.const_expr(cfg.ACTIVATION is not None):
                op = _ACTIVATIONS[cfg.ACTIVATION]

            if cutlass.const_expr(_is_packed16(cfg.DTYPE) and cfg.ACTIVATION is not None):
                kit_act = _packed16_kit(cfg.DTYPE)

            amax_r = Float32(0.0)
            for w in cutlass.range_constexpr(WAVES):
                idx = (w * PACK_SIZE + offset) % SCALE_DIM
                for e in cutlass.range_constexpr(PACK_SIZE):
                    x = Float32(sX_thread_rw[0, idx + e])
                    # If IS_ACT, apply activation function to x in f32
                    if cutlass.const_expr(cfg.ACTIVATION is not None):
                        # If it's relu, we can handle it later
                        if not cutlass.const_expr(FUSE_RELU):
                            x = op(x)
                    # If IS_DBIAS and not colwise, accumulate dbias from x in f32
                    if cutlass.const_expr(cfg.WITH_DBIAS and not cfg.COLWISE): 
                        j = w * PACK_SIZE + e
                        thread_dbias[j] = thread_dbias[j] + x
                    # If 16-bit input with activation, truncate to IType
                    if cutlass.const_expr(_is_packed16(cfg.DTYPE) and cfg.ACTIVATION is not None):
                        x = kit_act.truncate_f32(x) # TODO: Why not just qunatize from f32?
                    in_r[w][e] = x
                    if cutlass.const_expr(FUSE_RELU):
                        amax_r = cute.arch.fmax(amax_r, x) # For relu cases, we don't need abs since negative values will be 0 so they lose comparison automatically
                    else:
                        amax_r = cute.arch.fmax(amax_r, fabs_f32(x))
            if cutlass.const_expr(FUSE_RELU):
                amax_r = cute.arch.fmax(amax_r, Float32(0.0)) # If relu, the amax is at least 0

        # 2. E8M0 scale → gmem. mS_row's layout already encodes the swizzle
        # when cfg.WITH_GEMM_SWIZZLED_SCALES=True, so 2D access just works.
        biased_exp_r = float_to_e8m0(amax_r * max_norm_rcp)
        # mS_row_stage has logical shape (32, 2) and we have 64 threads where each is mapped to one scale factor
        # The TV layout is equivalent to https://kainzhong.github.io/CuTe-Layout-Visualizer/?key=tv-2-%2832%2C+2%29%3A%282%2C+1%29-%281%29
        # but it's too trival so let's just index it directly without using layout
        # Note this is the logical layout, which is on top of the swizzled / non-swizzled scale factor layout that mappes the logical index to the physical offset
        mS_row_stage[(tidx // 2, tidx % 2)] = Uint8(biased_exp_r)

        # 3. scale + packed fp8 cast → smem as one u32 per wave.
        inv_scale_r = exp2f_rcp(biased_exp_r) # f32 reciprocal of the scale
        # Fetch the conversion function based on the FP8 format
        cvt_f32x2 = _cvt_f32x2_to_fp8x2(cfg.FP8_DTYPE)
        if cutlass.const_expr(_row_fast):
            kit_cast = _packed16_kit(cfg.DTYPE)
            mul_cvt_x2 = kit_cast.mul_cvt_to_fp8x2(cfg.FP8_DTYPE, FUSE_RELU)
            # Pack `(inv_scale_r, inv_scale_r)` as a single 64-bit f32x2 once;
            # the per-wave mul_cvt consumes this directly.
            scale_2x = pack_f32x2(inv_scale_r, inv_scale_r)

        bank_group = (tidx % THREADS_PER_WARP) // THREADS_PER_BANK # Each 4 threads share the same bank, which forms a bank group
        offset = bank_group * 4 # Each bank group will write 4 fp8 to
        for w in cutlass.range_constexpr(WAVES):
            idx = (w * 4 + offset) % SCALE_DIM
            idx = idx // 4
            if cutlass.const_expr(_row_fast):
                # One fused PTX per <fmt>x2 pair: <fmt>x2 × f32x2 → fp8x2.
                # Byte layout: byte[0]=fp8(lo * s), byte[1]=fp8(hi * s).
                p01 = mul_cvt_x2(in_r[w][0], scale_2x)
                p23 = mul_cvt_x2(in_r[w][1], scale_2x)
            else:
                # cvt PTX semantics: `cvt.rn.satfinite.<fmt>.f32 d, a, b` gives
                # d[15:8]=fp8(a), d[7:0]=fp8(b). Pass (v1, v0) so the u16 low
                # byte ends up as fp8(v0) and the high byte as fp8(v1).
                v0 = in_r[w][0] * inv_scale_r
                v1 = in_r[w][1] * inv_scale_r
                v2 = in_r[w][2] * inv_scale_r
                v3 = in_r[w][3] * inv_scale_r
                p01 = cvt_f32x2(v1, v0, FUSE_RELU)  # u16 little-endian: v0,v1
                p23 = cvt_f32x2(v3, v2, FUSE_RELU)  # u16 little-endian: v2,v3
            quad = (p23 << Int32(16)) | p01
            sO_thread_u32[idx] = Uint32(quad)

        # Per-thread amax over the thread's 32-elt scale block. Also returns
        # the (possibly updated) thread_dbias accumulator — extended in the
        # path-B branch above; passed through unchanged otherwise.
        return amax_r, thread_dbias

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


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------
_compile_cache_tvm_ffi: dict = {}
_compile_cache: dict = {}

def _get_compiled_kernel(cfg, tvm_ffi=True):
    key = (cfg.DTYPE, cfg.FP8_DTYPE, cfg.ROWWISE, cfg.COLWISE,
           cfg.WITH_GEMM_SWIZZLED_SCALES, cfg.WITH_AMAX, cfg.ACTIVATION,
           cfg.WITH_DBIAS, "tvm_ffi" if tvm_ffi else "direct")
    cache = _compile_cache_tvm_ffi if tvm_ffi else _compile_cache
    if key not in cache:
        kernel_obj = MXFP8QuantizeSmemKernel(cfg)
        # kernel_obj = DoNothingKernel()
        # Sym dims. Reuse the SAME sym_int across fakes that share a logical
        # axis — tvm-ffi unifies dims by sym identity, so reusing M lets it
        # enforce e.g. mX.shape[0] == mO_row.shape[0] == mDbias is unrelated.
        # Conversely, use DIFFERENT sym_ints for axes that differ at runtime
        # (mS_row's N/32 vs mO_row's N), otherwise the wrapper will require
        # them to match and fail at call time.
        M = cute.sym_int()
        N = cute.sym_int()
        SCALED_N_ROW = cute.sym_int()   # mS_row dim 1 = N // 32
        SCALED_M_COL = cute.sym_int()   # mS_col dim 0 = M // 32
        WS_M = cute.sym_int()           # mDbias dim 0 = ceil_div(M, TILE_Y*NUM_TILES)
        # stride_order=(1, 0) → row-major: dim 1 is fastest-changing, stride 1.
        # Matches torch's default `torch.empty((M, N))` layout = strides (N, 1).
        # Default is (0, 1) (col-major-leftmost), which mismatches torch and
        # breaks the kernel's TMA layout assumptions internally.
        kw_rm16 = dict(stride_order=(1, 0),
                       memspace=cute.AddressSpace.gmem, assumed_align=16)
        kw_rm4 = dict(stride_order=(1, 0),
                      memspace=cute.AddressSpace.gmem, assumed_align=4)
        in_fake_ptr      = cute.runtime.make_fake_compact_tensor(cfg.DTYPE,  (M, N),            **kw_rm16)
        out_row_fake_ptr    = cute.runtime.make_fake_compact_tensor(cute.Uint8, (M, N),            **kw_rm16) if cfg.ROWWISE else None
        out_col_fake_ptr   = cute.runtime.make_fake_compact_tensor(cute.Uint8, (M, N),            **kw_rm16) if cfg.COLWISE else None
        scale_row_fake   = cute.runtime.make_fake_compact_tensor(cute.Uint8, (M, SCALED_N_ROW), **kw_rm16) if cfg.ROWWISE else None
        scale_col_fake   = cute.runtime.make_fake_compact_tensor(cute.Uint8, (SCALED_M_COL, N), **kw_rm16) if cfg.COLWISE else None
        ws_fake_ptr      = cute.runtime.make_fake_compact_tensor(Float32,    (WS_M, N),         **kw_rm4) if cfg.WITH_DBIAS else None
        single_fake_ptr  = dummy_cute_float32
        if tvm_ffi:
            compiled = cute.compile(
                kernel_obj,
                in_fake_ptr,                       # mX
                in_fake_ptr,                       # mActIn (alias of mX unless IS_DACT)
                out_row_fake_ptr,   scale_row_fake,   # mO_row, mS_row
                out_col_fake_ptr,   scale_col_fake,   # mO_col, mS_col
                single_fake_ptr,                   # mNoop
                single_fake_ptr,                   # mAmax
                ws_fake_ptr,                       # mDbias
                options="--enable-tvm-ffi"
            )
        else:
            compiled = cute.compile(
                kernel_obj,
                in_fake_ptr,                       # mX
                in_fake_ptr,                       # mActIn (alias of mX unless IS_DACT)
                out_row_fake_ptr,   scale_row_fake,   # mO_row, mS_row
                out_col_fake_ptr,   scale_col_fake,   # mO_col, mS_col
                single_fake_ptr,                   # mNoop
                single_fake_ptr,                   # mAmax
                ws_fake_ptr,                       # mDbias
            )
        cache[key] = compiled
    return cache[key]


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


def quantize_mxfp8_cutedsl(
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
) -> dict:
    """Quantize a 2D tensor to MXFP8 format using CuTeDSL kernels with smem tiling.

    Args:
        activation: forward fused activation ('relu'/'gelu'/'silu') or
            backward derivative ('drelu'/'dgelu'/'dsilu'), or None.
        act_input: REQUIRED when activation is a derivative. Holds the
            saved forward input `x` from the forward pass; the kernel
            evaluates `dOP(act_input)` and multiplies by `x` (the upstream
            grad). Same shape & dtype as `x`. Ignored otherwise.
        compute_dbias: when True, the kernel additionally computes the
            per-column sum of post-activation values (the bias gradient),
            returned as `result["dbias"]` of shape (N,) and dtype matching
            x.dtype. Currently requires `colwise=True` (only path A of the
            C++ kernel is implemented).
        noop: optional 1-element f32 cuda tensor. If `noop[0] == 1.0` at launch
            time, the kernel returns immediately and output buffers are left as
            allocated (uninitialised). Used for CUDA-Graph-friendly skip — see
            `swizzle_demo.svg`-adjacent docs / the C++ reference for semantics.
        with_amax: optional 1-element f32 cuda tensor. When supplied, the kernel
            atomic-maxes max(|x|) over the whole tensor into amax[0] (across
            all CTAs). Used by delayed-scaling FP8 modes that need a per-tensor
            amax alongside the per-32-element MXFP8 scales. Caller is
            responsible for initialising amax (e.g. to 0.0) before launch.
    """
    # t_start = time.perf_counter_ns()
    # print(f"Input tensor: shape={x.shape}, dtype={x.dtype}, device={x.device}")
    # nvtx = torch.cuda.nvtx
    # nvtx.range_push("dsl.validate")
    # assert x.is_cuda and x.is_contiguous() and x.ndim == 2
    # t0 = time.perf_counter_ns()
    M, N = x.shape
    # assert rowwise or colwise
    # assert M % TILE_Y == 0, f"M={M} must be a multiple of {TILE_Y}"
    # assert N % TILE_X == 0, f"N={N} must be a multiple of {TILE_X}"
    if with_gemm_swizzled_scales:
        # Swizzled tile is 128×4 in (M, N/32) → requires M and N to be
        # multiples of 128 to avoid partial-tile padding (which the host
        # would have to memset).
        # assert M % 128 == 0 and N % 128 == 0, (
        #     f"with_gemm_swizzled_scales requires M and N multiples of 128, "
        #     f"got M={M}, N={N}")
        pass
    if noop is None:
        noop = dummy_float32
    else:
        # assert noop.is_cuda and noop.dtype == torch.float32 and noop.numel() == 1, (
        #     f"noop must be a 1-element float32 cuda tensor, got "
        #     f"shape={tuple(noop.shape)}, dtype={noop.dtype}")
        pass
    if with_amax:
        # assert amax.is_cuda and amax.dtype == torch.float32 and amax.numel() == 1, (
        #     f"amax must be a 1-element float32 cuda tensor, got "
        #     f"shape={tuple(amax.shape)}, dtype={amax.dtype}")
        amax = torch.zeros(1, dtype=torch.float32, device="cuda")
        pass
    else:
        # Reuse the noop dummy slot — the kernel never reads/writes it when
        # cfg.WITH_AMAX is False, so any non-null pointer is fine.
        amax = dummy_float32
    if is_dact:
        # assert act_input is not None, (
        #     f"activation={activation!r} is a derivative — caller must pass "
        #     f"act_input (saved forward x).")
        # assert act_input.is_cuda and act_input.shape == x.shape and \
        #        act_input.dtype == x.dtype and act_input.is_contiguous(), (
        #     f"act_input must be a contiguous {x.dtype} cuda tensor with "
        #     f"shape={tuple(x.shape)}, got shape={tuple(act_input.shape)}, "
        #     f"dtype={act_input.dtype}")
        pass
    else:
        # Alias to x — the kernel ignores act_in_ptr in non-DACT paths, so
        # any valid (well-formed) pointer is fine. Aliasing avoids needing
        # a same-size dummy buffer.
        act_input = x
    if compute_dbias:
        # assert rowwise or colwise, (
        #     "compute_dbias=True requires rowwise or colwise to be True.")
        pass

    cutlass_dtype = _torch_to_cutlass_dtype[x.dtype]
    max_norm_rcp = FP8E4M3_MAX_NORM_RCP if fp8_dtype == "e4m3" else FP8E5M2_MAX_NORM_RCP
    # stream = torch.cuda.current_stream()
    # nvtx.range_pop()  # dsl.validate
    # t1 = time.perf_counter_ns()
    # timing_func("prepare_args", (t1 - t0) / 1e6)

    # t0 = time.perf_counter_ns()
    # nvtx.range_push("dsl.alloc")
    result = {}
    # if rowwise:
    #     result["rowwise_data"] = torch.empty((M, N), dtype=torch.uint8, device=x.device)
    #     result["rowwise_scale"] = torch.empty((M, N // SCALE_DIM), dtype=torch.uint8, device=x.device)
    # if colwise:
    #     result["colwise_data"] = torch.empty((M, N), dtype=torch.uint8, device=x.device)
    #     result["colwise_scale"] = torch.empty((M // SCALE_DIM, N), dtype=torch.uint8, device=x.device)
    # out_row_data  = result["rowwise_data"]  if rowwise else dummy_uint8
    # out_row_scale = result["rowwise_scale"] if rowwise else dummy_uint8
    # out_col_data  = result["colwise_data"]  if colwise else dummy_uint8
    # out_col_scale = result["colwise_scale"] if colwise else dummy_uint8

    # output_placeholder = tex.quantize(x, quantizer, None, noop_flag)
    # out_row_data = output_placeholder._rowwise_data if rowwise else dummy_uint8
    # out_row_scale = output_placeholder._rowwise_scale_inv if rowwise else dummy_uint8
    # out_col_data = output_placeholder._columnwise_data if colwise else dummy_uint8
    # out_col_scale = output_placeholder._columnwise_scale_inv if colwise else dummy_uint8
    out_row_data  = quantized_output._rowwise_data  if rowwise else None
    out_row_scale = quantized_output._rowwise_scale_inv if rowwise else None
    out_col_data  = quantized_output._colwise_data  if colwise else None
    out_col_scale = quantized_output._colwise_scale_inv if colwise else None
    # nvtx.range_pop()  # dsl.alloc
    # t1 = time.perf_counter_ns()
    # timing_func("allocate", (t1 - t0) / 1e6)

    # t0 = time.perf_counter_ns()
    # nvtx.range_push("dsl.cache_lookup")
    # Single unified kernel launch — loads global memory once for both directions
    cfg = MXFP8QuantizeConfig(cutlass_dtype, fp8_dtype, rowwise=rowwise, colwise=colwise,
                               with_gemm_swizzled_scales=with_gemm_swizzled_scales,
                               with_amax=with_amax, activation=activation,
                               with_dbias=compute_dbias, is_dact=is_dact)
    # compiled = _get_compiled_kernel_tvm_ffi(cfg)
    compiled = _get_compiled_kernel(cfg, True)
    # nvtx.range_pop()  # dsl.cache_lookup
    # t1 = time.perf_counter_ns()
    # timing_func("compile_kernel", (t1 - t0) / 1e6)

    # t0 = time.perf_counter_ns()
    # DBias workspace — `[blocks_Y, N]` f32 partial sums, reduced post-kernel.
    # blocks_Y must match the kernel's `(cfg.M + 63) // 64` (one CTA per
    # 64-row strip, since each CTA handles NUM_TILES=2 stages of TILE_Y=32).
    # nvtx.range_push("dsl.dbias_workspace")
    if compute_dbias:
        blocks_Y = (M + TILE_Y * NUM_TILES - 1) // (TILE_Y * NUM_TILES)
        dbias_workspace = torch.empty(
            (blocks_Y, N), dtype=torch.float32, device=x.device)
    else:
        # Fake declares (WS_M, N) — any positive WS_M is OK. Use a 1-row
        # dummy; the kernel never touches it when cfg.WITH_DBIAS is False.
        dbias_workspace = None
    # nvtx.range_pop()  # dsl.dbias_workspace
    # t1 = time.perf_counter_ns()
    # timing_func("dbias_workspace", (t1 - t0) / 1e6)

    # t0 = time.perf_counter_ns()
    # nvtx.range_push("dsl.launch")
    if not do_nothing:
        compiled(x, act_input,
            out_row_data, out_row_scale,
            out_col_data, out_col_scale,
            noop, amax, dbias_workspace)
    # nvtx.range_pop()  # dsl.launch
    # t1 = time.perf_counter_ns()
    # timing_func("launch", (t1 - t0) / 1e6)

    # t0 = time.perf_counter_ns()
    if compute_dbias:
        # Reduce blocks_Y partial sums along the block axis. torch.sum is
        # tree-based (single CUDA launch) — fast, but its association order
        # differs from TE's reduce_dbias kernel (sequential left-fold) so
        # fp32 dbias output drifts by ≤1 ULP. bf16/fp16 outputs round to
        # the same final byte. Workload-relevant trade: a sequential
        # Python-loop reduce would be bit-exact but launches `blocks_Y`
        # element-wise add kernels, dwarfing the quantize kernel time at
        # large shapes (e.g. 256 launches for M=16384).
        # nvtx.range_push("dsl.reduce_dbias")
        result["dbias"] = dbias_workspace.sum(dim=0).to(x.dtype)
        # nvtx.range_pop()  # dsl.reduce_dbias
    if with_amax:
        result["amax"] = amax[0].item()
    # t1 = time.perf_counter_ns()
    # timing_func("postprocess_output", (t1 - t0) / 1e6)
    # t_end = time.perf_counter_ns()
    # timing_func("total_time", (t_end - t_start) / 1e6)
    return result
