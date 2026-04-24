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
# Pin CuTeDSL compile target to Blackwell. Must be set before cutlass imports
# so env detection in base_dsl picks it up; also passed explicitly below.
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

from typing import Type

import cuda.bindings.driver as cuda
import torch

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

# MXFP8 settings
MXFP8_BLOCK_SIZE = 32   # alias used by tests
SCALE_DIM = 32           # Elements per MXFP8 scaling block (both dims)

# Double-buffering for async copy + compute overlap
BUFFER_NUM = 2

# Vectorised access constants for bank-conflict avoidance (rowwise pass)
PACK_SIZE = 4                              # Elements per vector load
WAVES = SCALE_DIM // PACK_SIZE             # 8 waves of 4 elements
THREADS_PER_WARP = 32
TOTAL_BANKS_WIDTH = (32 * 4) // 1  # 32 banks × 4 bytes, in bytes (uint8 stride)
THREADS_PER_BANK = TOTAL_BANKS_WIDTH // SCALE_DIM  # 4 threads per bank

NUM_STAGES = 2
NUM_TILES = 2
TILE_SIZE = 128
TILE_Y = 32
TILE_X = 64

# CTA size
THREADS_PER_CHUNK = 64

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
def abs_max_bf16x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """`max.xorsign.abs.bf16x2` — fused abs+max across two packed bf16 pairs.

    Takes two Int32s holding bf16x2 (low 16 bits = first bf16, high 16 bits =
    second), returns Int32 where each lane is max(|a|, |b|) in bf16. The sign
    of each output lane is `sign(a) XOR sign(b)` (the "xorsign" semantics) —
    callers must abs the final accumulator before interpreting the magnitude.
    """
    return Int32(llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "max.xorsign.abs.bf16x2 $0, $1, $2;",
        "=r,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def abs_max_bf16(a: Int16, b: Int16, *, loc=None, ip=None) -> Int16:
    """`max.xorsign.abs.bf16` — scalar-bf16 variant of the above."""
    return Int16(llvm.inline_asm(
        T.i16(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "max.xorsign.abs.bf16 $0, $1, $2;",
        "=h,h,h", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def bf16_bits_to_f32(bits: Int16, *, loc=None, ip=None) -> Float32:
    """Interpret 16 raw bits as bf16 and widen to f32 (place in top 16 of f32)."""
    i32 = Int32(mlir_arith.extui(
        T.i32(), bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
    return _bitcast_i32_to_f32(i32 << Int32(16), loc=loc, ip=ip)


@dsl_user_op
def bf16x2_lo_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
    """Extract the low bf16 lane of a packed bf16x2 Int32 and widen to f32."""
    return _bitcast_i32_to_f32(
        (bits & Int32(0xFFFF)) << Int32(16), loc=loc, ip=ip,
    )


@dsl_user_op
def bf16x2_hi_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
    """Extract the high bf16 lane of a packed bf16x2 Int32 and widen to f32."""
    # (x >> 16) << 16 is equivalent to `x & 0xFFFF0000` and sidesteps signed
    # literal issues. Arithmetic-right sign bits get zeroed by the left shift.
    return _bitcast_i32_to_f32(
        (bits >> Int32(16)) << Int32(16), loc=loc, ip=ip,
    )


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


@dsl_user_op
def mul_cvt_bf16x2_to_fp8e4m3x2(val_2x: Int32, scale_2x: Int64,
                                 *, loc=None, ip=None) -> Int32:
    """Fused `bf16x2 * f32x2 → fp8e4m3x2` — matches `ptx::mul_cvt_2x` in
    `transformer_engine/common/util/ptx.cuh`.

    - `val_2x`: packed bf16x2 (Int32), low 16 bits = element A, high = B.
    - `scale_2x`: packed f32x2 (Int64), low 32 = s_A, high 32 = s_B.
    - Returns Int32 whose low 16 bits are an fp8x2 in little-endian byte
      order: byte[0] = fp8(A * s_A), byte[1] = fp8(B * s_B). High 16 = 0.

    Internally unpacks the bf16x2, widens each half to f32, does a `mul.f32x2`
    with the scale pair, then a 2-wide satfinite cvt to e4m3x2 — all in one
    inline-asm block so the compiler keeps the intermediate values in regs.
    """
    result_i16 = Int16(llvm.inline_asm(
        T.i16(),
        [val_2x.ir_value(loc=loc, ip=ip), scale_2x.ir_value(loc=loc, ip=ip)],
        "{\n"
        ".reg.b64 vp0; .reg.b64 vp1;\n\t"
        ".reg.b32 v1;  .reg.b32 v2;\n\t"
        ".reg.b16 vb1; .reg.b16 vb2;\n\t"
        "mov.b32 {vb1, vb2}, $1;\n\t"
        "cvt.f32.bf16 v1, vb1;\n\t"
        "cvt.f32.bf16 v2, vb2;\n\t"
        "mov.b64 vp0, {v1, v2};\n\t"
        "mul.f32x2 vp1, vp0, $2;\n\t"
        "mov.b64 {v2, v1}, vp1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 $0, v1, v2;\n\t"
        "}",
        "=h,r,l", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
    return Int32(mlir_arith.extui(
        T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


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


@dsl_user_op
def gemm_swizzled_scale_idx(i: Int32, j: Int32, num_tiles_X: Int32,
                             *, loc=None, ip=None) -> Int32:
    """Convert compact scale indices (i, j) to GEMM-swizzled linear index.

    Matches swizzle.cuh::gemm_swizzled_scale_idx.
    Layout: 128×4 tiles, internal ordering per cuBLAS MXFP8 spec:
        (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4 + col_in_tile
    """
    TILE_DIM_X = Int32(4)
    TILE_DIM_Y = Int32(128)
    TILE_SIZE = Int32(512)   # 4 * 128

    tile_idx_X = j // TILE_DIM_X
    tile_idx_Y = i // TILE_DIM_Y
    idx_in_tile_X = j % TILE_DIM_X
    idx_in_tile_Y = i % TILE_DIM_Y

    idx = (tile_idx_Y * num_tiles_X + tile_idx_X) * TILE_SIZE
    idx = idx + (idx_in_tile_Y % Int32(32)) * Int32(16)
    idx = idx + (idx_in_tile_Y // Int32(32)) * Int32(4)
    idx = idx + idx_in_tile_X
    return idx


# ---------------------------------------------------------------------------
# Kernel configuration
# ---------------------------------------------------------------------------
class MXFP8QuantizeConfig:
    def __init__(self, dtype, M, N, fp8_dtype="e4m3", rowwise=True, colwise=False,
                 with_gemm_swizzled_scales=False):
        self.dtype = dtype
        self.M = M
        self.N = N
        self.fp8_dtype = fp8_dtype
        self.rowwise = rowwise
        self.colwise = colwise
        self.with_gemm_swizzled_scales = with_gemm_swizzled_scales
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
        scaling_type="rowwise", # "rowwise", "colwise", or "bidimensional"
    ):
        cfg = self.cfg
        num_scale_cols = cfg.N // SCALE_DIM
        num_scale_rows = cfg.M // SCALE_DIM

        mX = cute.make_tensor(x_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)))

        # Rowwise output tensors
        mO_row = cute.make_tensor(out_row_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)))
        if cutlass.const_expr(cfg.with_gemm_swizzled_scales):
            # Flat 1D scale tensor for swizzled writes
            mS_row = cute.make_tensor(
                scale_row_ptr,
                cute.make_layout((M * num_scale_cols,), stride=(1,)))
        else:
            mS_row = cute.make_tensor(
                scale_row_ptr,
                cute.make_layout((M, num_scale_cols), stride=(num_scale_cols, 1)))

        # Colwise output tensors
        mO_col = cute.make_tensor(out_col_ptr, cute.make_layout((M, cfg.N), stride=(cfg.N, 1)))
        if cutlass.const_expr(cfg.with_gemm_swizzled_scales):
            mS_col = cute.make_tensor(
                scale_col_ptr,
                cute.make_layout((num_scale_rows * cfg.N,), stride=(1,)))
        else:
            mS_col = cute.make_tensor(
                scale_col_ptr,
                cute.make_layout((num_scale_rows, cfg.N), stride=(cfg.N, 1)))
            
        # print(f"mX: {mX}\nmO_row: {mO_row}, mS_row: {mS_row}\nmO_col: {mO_col}, mS_col: {mS_col}\n")

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
        tma_atom_out_row, tma_dst_out_row = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_store, mO_row, out_smem_layout, cta_tiler, num_multicast=1,
        )
        tma_atom_out_col, tma_dst_out_col = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_store, mO_col, out_smem_layout, cta_tiler, num_multicast=1,
        )

        grid = [
            cute.ceil_div(Int32(cfg.N), TILE_X),
            cute.ceil_div(M, TILE_Y * NUM_TILES),
        ]
        block = [THREADS_PER_CHUNK,]

        self.kernel(
            mX, mO_row, mS_row, mO_col, mS_col,
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
        mX, # (M, N):(N, 1), the tensor to quantize
        mO_row, # (M, N):(N, 1), rowwise quantized output tensor (uint8)
        mS_row, # (M, N // 32):(N // 32, 1), rowwise scale tensor (uint8)
        mO_col, # (M, N):(N, 1), colwise quantized output tensor (uint8)
        mS_col, # (M // 32, N):(N, 1), colwise scale tensor (uint8)
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
        if cutlass.const_expr(cfg.rowwise and cfg.colwise):
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
        elif cutlass.const_expr(cfg.rowwise):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
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
        if cutlass.const_expr(cfg.rowwise):
            sO_row = storage.sO_row.get_tensor(
                cute.make_layout(
                    ((TILE_Y, TILE_X), NUM_STAGES),
                    stride=((TILE_X, 1), TILE_Y * TILE_X),
                )
            )
        if cutlass.const_expr(cfg.colwise):
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
        # tXsX: (TMA, NUM_STAGES)
        # tXgX: (TMA, (M/TILE_Y, N/TILE_X))
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            sX,
            gX_tiled,
        )

        # Same partitioning for S2G outputs: sO_row → mO_row and sO_col → mO_col.
        if cutlass.const_expr(cfg.rowwise):
            gO_row_tiled = cute.zipped_divide(tma_dst_out_row, (TILE_Y, TILE_X))
            tXsO_row, tXgO_row = cute.nvgpu.cpasync.tma_partition(
                tma_atom_out_row,
                0,
                cute.make_layout(1),
                sO_row,
                gO_row_tiled,
            )
        if cutlass.const_expr(cfg.colwise):
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

        # ---- Consumer: all threads quantize each completed tile. ----
        for stage in cutlass.range(num_tiles, unroll=1):
            mainloop_pipeline.consumer_wait(cons_state)
            sX_tile = sX[(None, cons_state.index)]          # (TILE_Y, TILE_X) bf16
            base_row = (bidy * NUM_TILES + stage) * TILE_Y

            if cutlass.const_expr(cfg.colwise):
                sO_col_tile = sO_col[(None, cons_state.index)]
                self._process_colwise(
                    sX_tile, sO_col_tile, base_row, bidx, tidx,
                    mS_col, max_norm_rcp,
                )
            if cutlass.const_expr(cfg.rowwise):
                sO_row_tile = sO_row[(None, cons_state.index)]
                self._process_rowwise(
                    sX_tile, sO_row_tile, base_row, bidx, tidx,
                    mS_row, max_norm_rcp,
                )

            # Make all smem stores (sO_row and/or sO_col) visible to the TMA
            # async proxy, then block-sync so warp 0 sees the fences from all
            # warps before issuing the bulk store(s). Matches the C++
            # reference's fence_proxy + __syncthreads pattern.
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            cute.arch.sync_threads()

            if warp_idx == 0:
                tile_y = bidy * NUM_TILES + stage
                if cutlass.const_expr(cfg.rowwise):
                    cute.copy(
                        tma_atom_out_row,
                        tXsO_row[(None, cons_state.index)],
                        tXgO_row[(None, (tile_y, bidx))],
                    )
                if cutlass.const_expr(cfg.colwise):
                    cute.copy(
                        tma_atom_out_col,
                        tXsO_col[(None, cons_state.index)],
                        tXgO_col[(None, (tile_y, bidx))],
                    )
                cute.arch.cp_async_bulk_commit_group()

            mainloop_pipeline.consumer_release(cons_state)
            cons_state.advance()

        # Wait for in-flight TMA stores so data is visible to the host
        # before the kernel returns.
        cute.arch.cp_async_bulk_wait_group(0, read=False)


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
        # For bf16 input we stash raw bf16 bits and run amax in bf16 via
        # `max.xorsign.abs.bf16`; otherwise fall back to the f32 path.
        if cutlass.const_expr(cfg.dtype is cutlass.BFloat16):
            sX_i16 = cute.make_tensor(
                cute.recast_ptr(sX_tile.iterator, dtype=Int16),
                cute.make_layout((TILE_Y, TILE_X), stride=(TILE_X, 1)),
            )
            in_c = [sX_i16[i, tidx] for i in range(SCALE_DIM)]

            # 1. bf16 amax — one PTX per element. Sign drifts (xorsign); take
            # fabs once at the end to recover the magnitude in f32.
            amax_bf16 = Int16(0)
            for i in cutlass.range_constexpr(SCALE_DIM):
                amax_bf16 = abs_max_bf16(amax_bf16, in_c[i])
            amax_c = fabs_f32(bf16_bits_to_f32(amax_bf16))
        else:
            in_c = [Float32(sX_flat[i, tidx]) for i in range(SCALE_DIM)]
            amax_c = Float32(0.0)
            for i in cutlass.range_constexpr(SCALE_DIM):
                amax_c = cute.arch.fmax(amax_c, fabs_f32(in_c[i]))

        # 2. E8M0 scale → gmem.
        biased_exp_c = float_to_e8m0(amax_c * max_norm_rcp)
        scale_row = base_row // SCALE_DIM
        if cutlass.const_expr(cfg.with_gemm_swizzled_scales):
            num_row_tiles = (cfg.M + 127) // 128
            sw_idx = gemm_swizzled_scale_idx(
                Int32(col_global), Int32(scale_row), Int32(num_row_tiles),
            )
            mS_col[sw_idx] = Uint8(biased_exp_c)
        else:
            mS_col[scale_row, col_global] = Uint8(biased_exp_c)

        # 3. scale + FP8 cast → smem (one byte per (row, tidx)). Caller
        # flushes the whole (TILE_Y, TILE_X) tile with a TMA S2G.
        inv_scale_c = exp2f_rcp(biased_exp_c)
        if cutlass.const_expr(cfg.dtype is cutlass.BFloat16):
            for i in cutlass.range_constexpr(SCALE_DIM):
                v_f32 = bf16_bits_to_f32(in_c[i])
                sO_col_flat[i, tidx] = Uint8(
                    cvt_f32_to_fp8e4m3(v_f32 * inv_scale_c)
                )
        else:
            for i in cutlass.range_constexpr(SCALE_DIM):
                sO_col_flat[i, tidx] = Uint8(
                    cvt_f32_to_fp8e4m3(in_c[i] * inv_scale_c)
                )

    @cute.jit
    def _process_rowwise(
        self,
        sX_tile,        # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
        sO_row_tile,    # (TILE_Y, TILE_X) uint8 smem view (rowwise FP8 output)
        base_row,       # Int32: global Y offset of this tile's first row
        bidx,           # Int32: block X index (column tile)
        tidx,           # Int32: thread index within the CTA
        mS_row,         # rowwise scale tensor (1D swizzled, or 2D linear)
        max_norm_rcp,
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

        # Match the C++ reference's thread layout: pairs of adjacent lanes
        # share a row (lanes 2k / 2k+1 both own row k), each pair covering
        # the two 32-element scale blocks of that row. Express as a cute
        # layout mapping `(tid_Y, tid_X) -> tidx` with stride (2, 1):
        # linear(tidx) = tid_Y*2 + tid_X, so `get_flat_coord` inverts to
        # `(tidx // 2, tidx % 2)` — semantically clearer than the raw
        # divmod, and readily reusable if we later partition via TiledCopy.
        rowwise_thread_layout = cute.make_layout((TILE_Y, 2), stride=(2, 1))
        tid_Y, tid_X = rowwise_thread_layout.get_flat_coord(tidx)

        # `bank_group` still has to key on the raw warp lane — each 4-thread
        # group shares a bank, independent of which rows those lanes own.
        bank_group = (tidx % THREADS_PER_WARP) // THREADS_PER_BANK

        global_row = base_row + tid_Y
        scale_col = bidx * 2 + tid_X
        col_base_local = tid_X * SCALE_DIM

        # Uint32 view of the rowwise output smem tile. Each wave (4 fp8
        # bytes) gets written as ONE st.shared.u32 instead of four u8
        # stores — matches the C++ reference's `Vec<OType2, 2>::store_to`.
        sO_u32_ptr = cute.recast_ptr(sO_row_tile.iterator, dtype=Uint32)
        sO_u32 = cute.make_tensor(
            sO_u32_ptr,
            cute.make_layout(
                (TILE_Y, TILE_X // 4), stride=(TILE_X // 4, 1),
            ),
        )

        if cutlass.const_expr(cfg.dtype is cutlass.BFloat16):
            # Read 4 consecutive bf16s per wave as TWO packed bf16x2 Int32s;
            # each ld.shared.b32 covers 2 bf16 elements. The cache `in_r[w][k]`
            # is an Int32 where low-half = element 2k, high-half = element 2k+1.
            sX_rw_i32 = cute.make_tensor(
                cute.recast_ptr(sX_tile.iterator, dtype=Int32),
                cute.make_layout(
                    (TILE_Y, 2, SCALE_DIM // 2),
                    stride=(TILE_X // 2, SCALE_DIM // 2, 1),
                ),
            )
            # 0. Load bf16x2 cache.
            in_r = [[None, None] for _ in range(WAVES)]
            for w in cutlass.range_constexpr(WAVES):
                swz = ((w + bank_group) * PACK_SIZE) % SCALE_DIM
                in_r[w][0] = sX_rw_i32[tid_Y, tid_X, swz // 2 + 0]
                in_r[w][1] = sX_rw_i32[tid_Y, tid_X, swz // 2 + 1]

            # 1. bf16x2 amax — 2 PTX per wave, 16 total per thread. Accumulates
            # `|elt|` in both lanes (with xorsign-drifted signs); final
            # horizontal max reduces the two bf16 lanes to a single f32.
            amax_2x = Int32(0)
            for w in cutlass.range_constexpr(WAVES):
                amax_2x = abs_max_bf16x2(amax_2x, in_r[w][0])
                amax_2x = abs_max_bf16x2(amax_2x, in_r[w][1])
            amax_r = cute.arch.fmax(
                fabs_f32(bf16x2_lo_to_f32(amax_2x)),
                fabs_f32(bf16x2_hi_to_f32(amax_2x)),
            )
        else:
            sX_rw = cute.make_tensor(
                sX_tile.iterator,
                cute.make_layout(
                    (TILE_Y, 2, SCALE_DIM),
                    stride=(TILE_X, SCALE_DIM, 1),
                ),
            )
            in_r = [[None] * PACK_SIZE for _ in range(WAVES)]
            for w in cutlass.range_constexpr(WAVES):
                swz = ((w + bank_group) * PACK_SIZE) % SCALE_DIM
                for e in cutlass.range_constexpr(PACK_SIZE):
                    in_r[w][e] = Float32(sX_rw[tid_Y, tid_X, swz + e])
            amax_r = Float32(0.0)
            for w in cutlass.range_constexpr(WAVES):
                for e in cutlass.range_constexpr(PACK_SIZE):
                    amax_r = cute.arch.fmax(amax_r, fabs_f32(in_r[w][e]))

        # 2. E8M0 scale → gmem.
        biased_exp_r = float_to_e8m0(amax_r * max_norm_rcp)
        if cutlass.const_expr(cfg.with_gemm_swizzled_scales):
            num_col_tiles = (cfg.N + 127) // 128
            sw_idx = gemm_swizzled_scale_idx(
                Int32(global_row), Int32(scale_col), Int32(num_col_tiles),
            )
            mS_row[sw_idx] = Uint8(biased_exp_r)
        else:
            mS_row[global_row, scale_col] = Uint8(biased_exp_r)

        # 3. scale + packed fp8 cast → smem as one u32 per wave.
        inv_scale_r = exp2f_rcp(biased_exp_r)
        if cutlass.const_expr(cfg.dtype is cutlass.BFloat16):
            # Pack `(inv_scale_r, inv_scale_r)` as a single 64-bit f32x2 once;
            # the per-wave mul_cvt consumes this directly.
            scale_2x = pack_f32x2(inv_scale_r, inv_scale_r)

        for w in cutlass.range_constexpr(WAVES):
            swz = ((w + bank_group) * PACK_SIZE) % SCALE_DIM
            if cutlass.const_expr(cfg.dtype is cutlass.BFloat16):
                # One fused PTX per bf16x2 pair: bf16x2 × f32x2 → fp8x2.
                # Byte layout: byte[0]=fp8(lo_bf16 * s), byte[1]=fp8(hi_bf16 * s).
                p01 = mul_cvt_bf16x2_to_fp8e4m3x2(in_r[w][0], scale_2x)
                p23 = mul_cvt_bf16x2_to_fp8e4m3x2(in_r[w][1], scale_2x)
            else:
                # cvt PTX semantics: `cvt.rn.satfinite.e4m3x2.f32 d, a, b` gives
                # d[15:8]=fp8(a), d[7:0]=fp8(b). Pass (v1, v0) so the u16 low
                # byte ends up as fp8(v0) and the high byte as fp8(v1).
                v0 = in_r[w][0] * inv_scale_r
                v1 = in_r[w][1] * inv_scale_r
                v2 = in_r[w][2] * inv_scale_r
                v3 = in_r[w][3] * inv_scale_r
                p01 = cvt_f32x2_to_fp8e4m3x2(v1, v0)  # u16 little-endian: v0,v1
                p23 = cvt_f32x2_to_fp8e4m3x2(v3, v2)  # u16 little-endian: v2,v3
            quad = (p23 << Int32(16)) | p01
            sO_u32[tid_Y, (col_base_local + swz) // 4] = Uint32(quad)


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------
_compile_cache: dict = {}


def _get_compiled_kernel(cfg, stream):
    key = (cfg.dtype, cfg.M, cfg.N, cfg.fp8_dtype, cfg.rowwise, cfg.colwise,
           cfg.with_gemm_swizzled_scales)
    if key not in _compile_cache:
        kernel_obj = MXFP8QuantizeSmemKernel(cfg)
        u8_ptr = make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
        compiled = cute.compile[(GPUArch("sm_100a"),)](
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
    with_gemm_swizzled_scales: bool = False,
) -> dict:
    """Quantize a 2D tensor to MXFP8 format using CuTeDSL kernels with smem tiling."""
    # print(f"Input tensor: shape={x.shape}, dtype={x.dtype}, device={x.device}")
    assert x.is_cuda and x.is_contiguous() and x.ndim == 2
    M, N = x.shape
    assert rowwise or colwise
    assert M % TILE_Y == 0, f"M={M} must be a multiple of {TILE_Y}"
    assert N % TILE_X == 0, f"N={N} must be a multiple of {TILE_X}"

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
    cfg = MXFP8QuantizeConfig(cutlass_dtype, M, N, fp8_dtype, rowwise=rowwise, colwise=colwise,
                               with_gemm_swizzled_scales=with_gemm_swizzled_scales)
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
