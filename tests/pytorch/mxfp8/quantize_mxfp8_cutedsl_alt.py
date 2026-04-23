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

from typing import Type

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, Int16, Uint8
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
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
TOTAL_BANKS_WIDTH = (32 * 4) / 1 # 32 banks, with 4 bytes each, divided by 1 byte per element (uint8)
THREADS_PER_BANK = TOTAL_BANKS_WIDTH // SCALE_DIM # (32 * 4) // 32 = 4 threads per bank

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
            
        print(f"mX: {mX}\nmO_row: {mO_row}, mS_row: {mS_row}\nmO_col: {mO_col}, mS_col: {mS_col}\n")

        # Declare TMA descriptors on the host side
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_tile_layout = cute.make_ordered_layout((TILE_Y, TILE_X), order=(1, 0))
        cta_tiler = (TILE_Y, TILE_X)
        gX = cute.zipped_divide(mX, cta_tiler)
        tma_atom, tma_src = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            gX,
            smem_tile_layout,
            cta_tiler,
            num_multicast=1,
        )
        print(f"tma_atom: {tma_atom}\ntma_src: {tma_src}\n")

        grid = [
            cute.ceil_div(Int32(cfg.N), TILE_X),
            cute.ceil_div(M, TILE_Y * NUM_TILES), 
        ]
        block = [THREADS_PER_CHUNK,]

        self.kernel(
            mX, mO_row, mS_row, mO_col, mS_col, 
            max_norm_rcp, mX.element_type,
            tma_atom, tma_src,
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
    ):

        @cute.struct
        class SharedStorage:
            mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
            sX: cute.struct.Align[
                cute.struct.MemRange[Float32, 32 * 64 * NUM_STAGES], 128
            ]
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        sX = storage.sX.get_tensor(cute.make_layout((32 * 64, NUM_STAGES,)))
        print(f"Shared memory buffer sX: {sX}\n")

        # 1 thread issues TMA to load the tile
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        # 64 threads consumes the tile to quantize
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS_PER_CHUNK)

        tx_count = TILE_SIZE * dtype.width // 8

        tma_pipe = pipeline.PipelineTmaAsync.create(
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

        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // 32)
        bidx, bidy, _ = cute.arch.block_idx()

        M = mX.shape[0]
        N = cfg.N

        num_tiles = cutlass.min(NUM_TILES, cute.ceil_div(M - bidy * TILE_Y * NUM_TILES, TILE_Y) )

        # Prologue: warp 0 prefetches the first NUM_STAGES-1 tiles.
        if warp_idx == 0: # In CuTeDSL the leader election is performed inside the atom
            prefetch_count = NUM_STAGES - 1
            for k in cutlass.range(prefetch_count, unroll=1):
                tma_pipe.producer_acquire(prod_state)
                cute.copy(
                    tma_atom,
                    tma_src[(None, k)],
                    sX[(None, prod_state.index)],
                    tma_bar_ptr=tma_pipe.producer_get_barrier(prod_state),
                )
                prod_state.advance()

        # Consumers reads from SMEM
        for k in cutlass.range(num_tiles, unroll=1):
            tma_pipe.consumer_wait(cons_state)

            sX_tile = sX[(None, cons_state.index)]
            
            self._process_tile(sX_tile, mO_row, mS_row, mO_col, mS_col, max_norm_rcp)

            tma_pipe.consumer_release(cons_state)
            cons_state.advance()

            if tidx == 0:
                k_prefetch = k + (NUM_STAGES - 1)
                if k_prefetch < num_tiles:
                    tma_pipe.producer_acquire(prod_state)
                    cute.copy(
                        tma_atom,
                        tma_src[(None, k_prefetch)],
                        sX[(None, prod_state.index)],
                        tma_bar_ptr=tma_pipe.producer_get_barrier(prod_state),
                    )
                    prod_state.advance()


    @cute.jit
    def _process_tile(
        self,
        sX_tile,
        mO_row, # Quantized output tensor for rowwise pass (uint8)
        mS_row, # Scale tensor for rowwise pass (uint8)
        mO_col, # Quantized output tensor for colwise pass (uint8)
        mS_col, # Scale tensor for colwise pass (uint8)
        max_norm_rcp,
    ):
        print(f"sX_tile: {sX_tile}")
        # block_off_Y = bidy * CHUNK_DIM_Y
        # block_off_X = bidx * CHUNK_DIM_X

        # # Thread mappings
        # tid_Y_row = tidx // THREADS_X
        # tid_X_row = tidx % THREADS_X
        # g_col_cw = block_off_X + tidx

        # # Bank-conflict constants (used by rowwise pass)
        # thread_lane = tidx % THREADS_PER_WARP
        # bank_group = thread_lane // THREADS_PER_BANK

        # # Output smem buffers (FP8 = uint8, 32×64 = 2KB each)
        # out_smem_layout = cute.make_ordered_layout(
        #     (BUFF_DIM_Y, BUFF_DIM_X), order=(1, 0))
        # if cutlass.const_expr(cfg.rowwise):
        #     sO_row = smem.allocate_tensor(Uint8, out_smem_layout, byte_alignment=128)
        # if cutlass.const_expr(cfg.colwise):
        #     sO_col = smem.allocate_tensor(Uint8, out_smem_layout, byte_alignment=128)

        # # --- cp.async TiledCopy: 64 threads, 128-bit (8-element) vectorised ---
        # # thr (8,8): 8 thread-rows × 8 thread-cols = 64 threads
        # # val (4,8): 4 rows × 8 elements per thread = 32 elements/thread
        # # → tile = (32, 64)
        # thr_layout = cute.make_layout((2,32), stride=(32,1))
        # val_layout = cute.make_layout((16,2), stride=(2,1))
        # # https://kainzhong.github.io/CuTe-Layout-Visualizer/?key=tv-2-%288%2C8%29%3A%288%2C1%29-%284%2C8%29%3A%288%2C1%29
        # print(f"thr_layout: {thr_layout}, val_layout: {val_layout}, make_layout_tv: {cute.make_layout_tv(thr_layout, val_layout)}\n")
        # copy_atom_async = cute.make_copy_atom(
        #     cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type,
        #     num_bits_per_copy=32,
        # )
        # tiled_copy = cute.make_tiled_copy_tv(
        #     copy_atom_async, thr_layout, val_layout)
        # print(f"copy_atom_async: {copy_atom_async}\ntiled_copy: {tiled_copy}\n")
        # thr_copy = tiled_copy.get_slice(tidx)
        # print(f"thr_copy: {thr_copy}")

        # # Per-stage global tiles: local_tile with (32, 64) tiler
        # tiler = (BUFF_DIM_Y, BUFF_DIM_X)
        # print(f"tiler: {tiler}")
        # gX_s0 = cute.local_tile(mX, tiler, (bidy * 2, bidx))
        # gX_s1 = cute.local_tile(mX, tiler, (bidy * 2 + 1, bidx))
        # print(f"mx: {mX}\ngX_s0: {gX_s0}\ngX_s1: {gX_s1}\n")

        # # Partition global and smem for each stage
        # tXgX_s0 = thr_copy.partition_S(gX_s0)
        # tXsX_s0 = thr_copy.partition_D(sX0)
        # tXgX_s1 = thr_copy.partition_S(gX_s1)
        # tXsX_s1 = thr_copy.partition_D(sX1)
        # print(f"tXgX_s0: {tXgX_s0}, tXsX_s0: {tXsX_s0}, tXgX_s1: {tXgX_s1}, tXsX_s1: {tXsX_s1}\n")

        # # --- Issue both cp.async loads upfront for maximum overlap ---
        # cute.copy(copy_atom_async, tXgX_s0, tXsX_s0)
        # cute.arch.cp_async_commit_group()
        # cute.copy(copy_atom_async, tXgX_s1, tXsX_s1)
        # cute.arch.cp_async_commit_group()

        # # Output smem refs (use sO_row / sO_col allocated above; None when disabled)
        # sO_row_ref = sO_row if cutlass.const_expr(cfg.rowwise) else None
        # sO_col_ref = sO_col if cutlass.const_expr(cfg.colwise) else None

        # # --- Stage 0: wait for buf 0, compute (stage 1 loads in background) ---
        # cute.arch.cp_async_wait_group(1)
        # cute.arch.barrier()

        # base_row_0 = block_off_Y
        # if cutlass.const_expr(cfg.colwise):
        #     self._compute_stage_colwise(
        #         sX0, sO_col_ref, mS_col,
        #         base_row_0, block_off_X, g_col_cw,
        #         max_norm_rcp,
        #     )
        # if cutlass.const_expr(cfg.rowwise):
        #     self._compute_stage_rowwise(
        #         sX0, sO_row_ref, mS_row,
        #         base_row_0, bidx, tid_Y_row, tid_X_row,
        #         bank_group, max_norm_rcp,
        #     )
        # # Flush output smem → global (coalesced cooperative store)
        # cute.arch.barrier()
        # self._flush_output_smem(
        #     sO_row_ref, sO_col_ref, mO_row, mO_col,
        #     base_row_0, block_off_X, tidx,
        # )

        # # --- Stage 1: wait for buf 1, compute ---
        # cute.arch.cp_async_wait_group(0)
        # cute.arch.barrier()

        # base_row_1 = block_off_Y + BUFF_DIM_Y
        # if cutlass.const_expr(cfg.colwise):
        #     self._compute_stage_colwise(
        #         sX1, sO_col_ref, mS_col,
        #         base_row_1, block_off_X, g_col_cw,
        #         max_norm_rcp,
        #     )
        # if cutlass.const_expr(cfg.rowwise):
        #     self._compute_stage_rowwise(
        #         sX1, sO_row_ref, mS_row,
        #         base_row_1, bidx, tid_Y_row, tid_X_row,
        #         bank_group, max_norm_rcp,
        #     )
        # cute.arch.barrier()
        # self._flush_output_smem(
        #     sO_row_ref, sO_col_ref, mO_row, mO_col,
        #     base_row_1, block_off_X, tidx,
        # )

    @cute.jit
    def _compute_stage_colwise(
        self, sX, sO_col, mS_col,
        base_row, block_off_X, g_col_cw,
        max_norm_rcp,
    ):
        """Colwise pass: thread `tidx` owns one column of the 32×64 tile.

        Reads 32 elements down the column, computes the MXFP8 E8M0 scale,
        scales + casts to FP8, stores to output smem and the scale to gmem.
        """
        cfg = self.cfg
        tidx_local = g_col_cw - block_off_X
        scale_row = base_row // SCALE_DIM

        # 1. Amax over the 32-element column
        amax_c = Float32(0.0)
        for i in cutlass.range_constexpr(SCALE_DIM):
            val = Float32(sX[i, tidx_local])
            amax_c = cute.arch.fmax(amax_c, fabs_f32(val))

        # 2. E8M0 scale → gmem (scalar, scattered)
        biased_exp_c = float_to_e8m0(amax_c * max_norm_rcp)
        if cutlass.const_expr(cfg.with_gemm_swizzled_scales):
            num_row_tiles = (cfg.M + 127) // 128
            sw_idx = gemm_swizzled_scale_idx(g_col_cw, Int32(scale_row),
                                              Int32(num_row_tiles))
            mS_col[sw_idx] = Uint8(biased_exp_c)
        else:
            mS_col[scale_row, g_col_cw] = Uint8(biased_exp_c)

        # 3. Scale + FP8 cast → output smem
        inv_scale_c = exp2f_rcp(biased_exp_c)
        for i in cutlass.range_constexpr(SCALE_DIM):
            val = Float32(sX[i, tidx_local])
            sO_col[i, tidx_local] = Uint8(cvt_f32_to_fp8e4m3(val * inv_scale_c))

    @cute.jit
    def _compute_stage_rowwise(
        self, sX, sO_row, mS_row,
        base_row, bidx, tid_Y_row, tid_X_row,
        bank_group, max_norm_rcp,
    ):
        """Rowwise pass: thread (tid_Y, tid_X) owns one 32-element scale block.

        Reads 8 waves × 4 elements with bank-group swizzle to avoid smem bank
        conflicts.  Uses 2-wide FP8 conversion (cvt.rn.satfinite.e4m3x2.f32).
        """
        cfg = self.cfg
        g_row_rw  = base_row + tid_Y_row
        scale_col = bidx * THREADS_X + tid_X_row
        col_start = tid_X_row * SCALE_DIM

        # 1. Amax over 8 waves × 4 elements (bank-conflict-free order)
        amax_r = Float32(0.0)
        for w in cutlass.range_constexpr(WAVES):
            swizzled_grp = ((w + bank_group) * PACK_SIZE) % SCALE_DIM
            for e in cutlass.range_constexpr(PACK_SIZE):
                val = Float32(sX[tid_Y_row, col_start + swizzled_grp + e])
                amax_r = cute.arch.fmax(amax_r, fabs_f32(val))

        # 2. E8M0 scale → gmem
        biased_exp_r = float_to_e8m0(amax_r * max_norm_rcp)
        if cutlass.const_expr(cfg.with_gemm_swizzled_scales):
            num_col_tiles = (cfg.N + 127) // 128
            sw_idx = gemm_swizzled_scale_idx(g_row_rw, scale_col,
                                              Int32(num_col_tiles))
            mS_row[sw_idx] = Uint8(biased_exp_r)
        else:
            mS_row[g_row_rw, scale_col] = Uint8(biased_exp_r)

        # 3. Scale + 2-wide FP8 cast → output smem (swizzled writes)
        inv_scale_r = exp2f_rcp(biased_exp_r)
        for w in cutlass.range_constexpr(WAVES):
            swizzled_grp = ((w + bank_group) * PACK_SIZE) % SCALE_DIM
            for e in cutlass.range_constexpr(PACK_SIZE // 2):
                v0 = Float32(sX[tid_Y_row, col_start + swizzled_grp + 2 * e])
                v1 = Float32(sX[tid_Y_row, col_start + swizzled_grp + 2 * e + 1])
                packed = cvt_f32x2_to_fp8e4m3x2(v0 * inv_scale_r,
                                                 v1 * inv_scale_r)
                sO_row[tid_Y_row, col_start + swizzled_grp + 2 * e] = Uint8(
                    packed >> Int32(8))
                sO_row[tid_Y_row, col_start + swizzled_grp + 2 * e + 1] = Uint8(
                    packed & Int32(0xFF))

    @cute.jit
    def _flush_output_smem(
        self, sO_row, sO_col, mO_row, mO_col,
        base_row, block_off_X, tidx,
    ):
        """Coalesced smem→gmem flush via 128-bit vectorized stores.

        Lowers to `st.global.v4.b32` (STG.E.128): 16 B/thread per atom.
        Tile = 32×64 uint8 = 2048 B; 64 threads × 32 B = 2 atoms/thread.
        Thread grid (16,4), val (2,16) → each thread writes 16 contiguous
        bytes in each of 2 rows.
        """
        cfg = self.cfg

        thr_layout = cute.make_layout((16, 4), stride=(4, 1))
        val_layout = cute.make_layout((2, 16), stride=(16, 1))
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Uint8,
            num_bits_per_copy=128,
        )
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)

        tiler = (BUFF_DIM_Y, BUFF_DIM_X)
        tile_y = base_row // BUFF_DIM_Y
        tile_x = block_off_X // BUFF_DIM_X

        if cutlass.const_expr(cfg.rowwise):
            gO_row = cute.local_tile(mO_row, tiler, (tile_y, tile_x))
            cute.copy(copy_atom,
                      thr_copy.partition_S(sO_row),
                      thr_copy.partition_D(gO_row))

        if cutlass.const_expr(cfg.colwise):
            gO_col = cute.local_tile(mO_col, tiler, (tile_y, tile_x))
            cute.copy(copy_atom,
                      thr_copy.partition_S(sO_col),
                      thr_copy.partition_D(gO_col))


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
    with_gemm_swizzled_scales: bool = False,
) -> dict:
    """Quantize a 2D tensor to MXFP8 format using CuTeDSL kernels with smem tiling."""
    assert x.is_cuda and x.is_contiguous() and x.ndim == 2
    M, N = x.shape
    assert rowwise or colwise
    # assert M % CHUNK_DIM_Y == 0, f"M={M} must be a multiple of {CHUNK_DIM_Y}"
    # assert N % CHUNK_DIM_X == 0, f"N={N} must be a multiple of {CHUNK_DIM_X}"

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
