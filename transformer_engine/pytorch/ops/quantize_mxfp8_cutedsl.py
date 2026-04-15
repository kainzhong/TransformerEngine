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

# Vectorised access constants for bank-conflict avoidance (rowwise pass)
PACK_SIZE = 4                              # Elements per vector load
WAVES = SCALE_DIM // PACK_SIZE             # 8 waves of 4 elements
THREADS_PER_WARP = 32
THREADS_PER_BANK = (32 * 4) // SCALE_DIM  # 4 — threads sharing a bank group

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
def cvt_f32x2_to_fp8e4m3x2(val_hi: Float32, val_lo: Float32,
                             *, loc=None, ip=None) -> Int32:
    """Convert two float32 values to two packed fp8e4m3fn bytes in one instruction.

    Returns an int32 where bits [7:0] = fp8(val_lo), bits [15:8] = fp8(val_hi).
    This mirrors ptx::mul_cvt_2x which converts 2 values in one instruction.
    """
    result_i16 = Int16(llvm.inline_asm(
        T.i16(),
        [val_hi.ir_value(loc=loc, ip=ip), val_lo.ir_value(loc=loc, ip=ip)],
        "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
        "=h,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
    return Int32(mlir_arith.extui(
        T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


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
        x_ptr,
        out_row_ptr, scale_row_ptr,
        out_col_ptr, scale_col_ptr,
        M, max_norm_rcp, stream,
    ):
        cfg = self.cfg
        num_scale_cols = cfg.N // SCALE_DIM
        num_scale_rows = cfg.M // SCALE_DIM

        mX = cute.make_tensor(x_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)))

        # Rowwise output tensors
        mO_row = cute.make_tensor(out_row_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)))
        mS_row = cute.make_tensor(
            scale_row_ptr,
            cute.make_layout((M, num_scale_cols), stride=(num_scale_cols, 1)))

        # Colwise output tensors
        mO_col = cute.make_tensor(out_col_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)))
        mS_col = cute.make_tensor(
            scale_col_ptr,
            cute.make_layout((num_scale_rows, cfg.N), stride=(cfg.N, 1)))

        self.kernel(mX, mO_row, mS_row, mO_col, mS_col, max_norm_rcp).launch(
            grid=[cute.ceil_div(Int32(cfg.N), CHUNK_DIM_X),
                  cute.ceil_div(M, CHUNK_DIM_Y), 1],
            block=[THREADS_PER_CHUNK, 1, 1],
            smem=2 * BUFF_DIM_Y * BUFF_DIM_X * (cfg.dtype.width // 8) + 256,
        )

    @cute.kernel
    def kernel(self, mX, mO_row, mS_row, mO_col, mS_col, max_norm_rcp):
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        M = mX.shape[0]
        N = cfg.N

        block_off_Y = bidy * CHUNK_DIM_Y
        block_off_X = bidx * CHUNK_DIM_X

        # Thread mappings
        tid_Y_row = tidx // THREADS_X
        tid_X_row = tidx % THREADS_X
        g_col_cw = block_off_X + tidx   # colwise: thread per column

        # --- Double-buffered shared memory (BUFFS_NUM = 2) ---
        smem = utils.SmemAllocator()
        sX0 = smem.allocate_tensor(
            cfg.dtype,
            cute.make_ordered_layout((BUFF_DIM_Y, BUFF_DIM_X), order=(1, 0)),
            byte_alignment=128,
        )
        sX1 = smem.allocate_tensor(
            cfg.dtype,
            cute.make_ordered_layout((BUFF_DIM_Y, BUFF_DIM_X), order=(1, 0)),
            byte_alignment=128,
        )

        # Bank-conflict constants (used by rowwise pass)
        thread_lane = tidx % THREADS_PER_WARP
        bank_group = thread_lane // THREADS_PER_BANK

        # --- Prefetch stage 0 into buffer 0 ---
        base_row_0 = block_off_Y
        for r in cutlass.range_constexpr(BUFF_DIM_Y):
            sX0[r, tidx] = mX[base_row_0 + r, block_off_X + tidx]
        cute.arch.barrier()

        # --- Stage 0: compute from buf 0, prefetch stage 1 into buf 1 ---
        self._compute_stage(
            sX0, mO_row, mS_row, mO_col, mS_col,
            base_row_0, block_off_X, bidx,
            tid_Y_row, tid_X_row, g_col_cw,
            bank_group, max_norm_rcp,
        )

        # Prefetch stage 1 into buffer 1 (overlaps with stage 0 global writes)
        base_row_1 = block_off_Y + BUFF_DIM_Y
        cute.arch.barrier()
        for r in cutlass.range_constexpr(BUFF_DIM_Y):
            sX1[r, tidx] = mX[base_row_1 + r, block_off_X + tidx]
        cute.arch.barrier()

        # --- Stage 1: compute from buf 1 ---
        self._compute_stage(
            sX1, mO_row, mS_row, mO_col, mS_col,
            base_row_1, block_off_X, bidx,
            tid_Y_row, tid_X_row, g_col_cw,
            bank_group, max_norm_rcp,
        )

    @cute.jit
    def _compute_stage(
        self, sX, mO_row, mS_row, mO_col, mS_col,
        base_row, block_off_X, bidx,
        tid_Y_row, tid_X_row, g_col_cw,
        bank_group, max_norm_rcp,
    ):
        """Process one 32×64 tile from shared memory (colwise then rowwise)."""
        cfg = self.cfg

        # --- Colwise pass ---
        if cutlass.const_expr(cfg.colwise):
            scale_row = base_row // SCALE_DIM

            amax_c = Float32(0.0)
            for i in cutlass.range_constexpr(SCALE_DIM):
                tidx_local = g_col_cw - block_off_X
                val = Float32(sX[i, tidx_local])
                amax_c = cute.arch.fmax(amax_c, fabs_f32(val))

            biased_exp_c = float_to_e8m0(amax_c * max_norm_rcp)
            mS_col[scale_row, g_col_cw] = Uint8(biased_exp_c)

            inv_scale_c = exp2f_rcp(biased_exp_c)
            for i in cutlass.range_constexpr(SCALE_DIM):
                tidx_local = g_col_cw - block_off_X
                val = Float32(sX[i, tidx_local])
                mO_col[base_row + i, g_col_cw] = Uint8(
                    cvt_f32_to_fp8e4m3(val * inv_scale_c))

        # --- Rowwise pass — bank-conflict-free wave access ---
        if cutlass.const_expr(cfg.rowwise):
            g_row_rw = base_row + tid_Y_row
            scale_col = bidx * THREADS_X + tid_X_row
            col_start = tid_X_row * SCALE_DIM

            amax_r = Float32(0.0)
            for w in cutlass.range_constexpr(WAVES):
                swizzled_grp = ((w + bank_group) * PACK_SIZE) % SCALE_DIM
                for e in cutlass.range_constexpr(PACK_SIZE):
                    val = Float32(sX[tid_Y_row, col_start + swizzled_grp + e])
                    amax_r = cute.arch.fmax(amax_r, fabs_f32(val))

            biased_exp_r = float_to_e8m0(amax_r * max_norm_rcp)
            mS_row[g_row_rw, scale_col] = Uint8(biased_exp_r)

            inv_scale_r = exp2f_rcp(biased_exp_r)
            g_col_base = block_off_X + col_start
            for w in cutlass.range_constexpr(WAVES):
                swizzled_grp = ((w + bank_group) * PACK_SIZE) % SCALE_DIM
                for e in cutlass.range_constexpr(PACK_SIZE // 2):
                    v0 = Float32(sX[tid_Y_row, col_start + swizzled_grp + 2 * e])
                    v1 = Float32(sX[tid_Y_row, col_start + swizzled_grp + 2 * e + 1])
                    packed = cvt_f32x2_to_fp8e4m3x2(v0 * inv_scale_r,
                                                     v1 * inv_scale_r)
                    g_c = g_col_base + swizzled_grp + 2 * e
                    mO_row[g_row_rw, g_c] = Uint8(packed >> Int32(8))
                    mO_row[g_row_rw, g_c + 1] = Uint8(packed & Int32(0xFF))


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------
_compile_cache: dict = {}


def _get_compiled_kernel(cfg, stream):
    key = (cfg.dtype, cfg.M, cfg.N, cfg.fp8_dtype, cfg.rowwise, cfg.colwise)
    if key not in _compile_cache:
        kernel_obj = MXFP8QuantizeSmemKernel(cfg)
        u8_ptr = make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
        compiled = cute.compile(
            kernel_obj,
            make_ptr(cfg.dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
            u8_ptr, u8_ptr,   # rowwise data, scale
            u8_ptr, u8_ptr,   # colwise data, scale
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

    # Allocate outputs
    result = {}
    if rowwise:
        result["rowwise_data"] = torch.empty((M, N), dtype=torch.uint8, device=x.device)
        result["rowwise_scale"] = torch.empty((M, N // SCALE_DIM), dtype=torch.uint8, device=x.device)
    if colwise:
        result["colwise_data"] = torch.empty((M, N), dtype=torch.uint8, device=x.device)
        result["colwise_scale"] = torch.empty((M // SCALE_DIM, N), dtype=torch.uint8, device=x.device)

    # Single unified kernel launch — loads global memory once for both directions
    cfg = MXFP8QuantizeConfig(cutlass_dtype, M, N, fp8_dtype, rowwise=rowwise, colwise=colwise)
    compiled = _get_compiled_kernel(cfg, stream)

    # For unused directions, point to the other direction's buffer (never written)
    dummy = result.get("rowwise_data", result.get("colwise_data"))
    dummy_scale = result.get("rowwise_scale", result.get("colwise_scale"))

    def _ptr(t):
        return make_ptr(Uint8, t.data_ptr())

    compiled(
        make_ptr(cutlass_dtype, x.data_ptr()),
        _ptr(result["rowwise_data"]) if rowwise else _ptr(dummy),
        _ptr(result["rowwise_scale"]) if rowwise else _ptr(dummy_scale),
        _ptr(result["colwise_data"]) if colwise else _ptr(dummy),
        _ptr(result["colwise_scale"]) if colwise else _ptr(dummy_scale),
        Int32(M), Float32(max_norm_rcp), stream,
    )

    return result
