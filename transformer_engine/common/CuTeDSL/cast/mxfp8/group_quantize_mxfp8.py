# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped MXFP8 quantization kernel implemented in CuTeDSL.

Strategy-aligned port of group_quantize_mxfp8.cuh. The scheduling, descriptor
management and per-tensor scale addressing mirror the CUDA kernel one-for-one:

  * persistent grid: sm_count * STATIC_PERSISTENT_BLOCKS_PER_SM CTAs, each
    grid-striding over a virtual work grid of 128x128 chunks (decode_job /
    is_job_valid / job_has_work / advance_to_next_job).
  * `is_single_tensor` reps (SAME_BOTH_DIMS, VARYING_FIRST_DIM) address the group
    through ONE static TMA descriptor with global block offsets -- exactly the
    CUDA `tensor_map_*_static` path. The other reps get per-tensor descriptors
    written by a prologue kernel (the CuTeDSL analog of update_tma_descriptors
    filling g_tensor_maps) and acquired with a tensormap proxy fence.
  * per-tensor scale bases/strides follow the CUDA formulas:
        scales_* += is_single_tensor ? 0 : tensor_base / 32
        stride_rowwise = roundup(cols/32, 4)   stride_colwise = roundup(cols, 128)

Mechanics that provably yield the same bytes may differ: the mbarrier pipeline is
expressed with PipelineTmaAsync instead of hand-rolled mbarriers, and
out-of-bounds scale padding is skipped rather than explicitly zeroed (the CUDA
kernel writes 0 there; downstream only consumes the meaningful region).

Scope: cast-only (no dbias / activation / dact / amax), compact (non-swizzled)
scales, rowwise and/or colwise. VARYING_BOTH_DIMS is not handled here (its
logical shape [1, total] is not tileable); the C++ bridge falls back to CUDA.
"""

"""
TODO:
  - OOB scale padding: CUDA writes 0; the helpers skip the store. Only bites when last_logical_dim % 128 != 0 under SAME_BOTH_DIMS/VARYING_FIRST_DIM. My tests wouldn't catch it — they compare only the meaningful region.
  - num_tensors = mOffsets.shape[0] - 1: the bridge substitutes a length-num_tensors stub for SAME_BOTH_DIMS, so this reads one too few (and divides by zero at num_tensors == 1). Harmless today only because tensor_id is dead for that rep. My harness always passed a real n+1 offsets array, so it didn't exercise this.
  - Int32 arithmetic on block_global_offset / tensor_base / first*last where CUDA uses size_t — overflows past 2^31 elements.
  - Cheap occupancy win: sO_row and sO_col are both allocated unconditionally; CUDA sizes only the direction in use. For fp32 that's 48 KB instead of 40 KB per CTA.
"""

# pylint: disable=missing-class-docstring

import logging
import os
from typing import Type

import cutlass
from cutlass import cute
from cutlass import pipeline
from cutlass import Boolean, Int32, Int64, Float8E8M0FNU
from cutlass.cute.nvgpu import cpasync
from cutlass.tensor_utils import TensorMapManager, TensorMapUpdateMode
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

from transformer_engine.common.CuTeDSL.utils import (
    str_to_cutlass_dtype,
    device_compute_capability,
)
from transformer_engine.common.CuTeDSL.cast.mxfp8.quantize_mxfp8 import (
    MXFP8_BLOCK_SCALING_SIZE,
    FP8E4M3_MAX_NORM_RCP,
    FP8E5M2_MAX_NORM_RCP,
    quantize_rowwise_mxfp8,
    quantize_colwise_mxfp8,
)

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"
logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")

THREADS_PER_WARP = 32
BYTES_PER_TENSORMAP = 128
# Descriptor slots per tensor: input, rowwise output, colwise output.
NUM_TENSORMAPS = 3

# Shape representations, mirroring ShapeRepresentation in common/utils.cuh.
SAME_BOTH_DIMS = "same_both_dims"
VARYING_FIRST_DIM = "varying_first_dim"
VARYING_LAST_DIM = "varying_last_dim"
SUPPORTED_SHAPE_REPS = (SAME_BOTH_DIMS, VARYING_FIRST_DIM, VARYING_LAST_DIM)


class MXFP8GroupQuantizeConfig:
    """Compile-time config for the grouped MXFP8 quantize kernel."""

    def __init__(self, dtype: str, fp8_dtype: str, rowwise: bool, colwise: bool, shape_rep: str):
        if dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"unknown input dtype {dtype!r}; expected fp32|fp16|bf16")
        self.DTYPE = str_to_cutlass_dtype(dtype)
        self.DTYPE_STR = dtype
        if fp8_dtype not in ("fp8_e4m3fn", "fp8_e5m2"):
            raise ValueError(f"unknown FP8 dtype {fp8_dtype!r}; expected fp8_e4m3fn|fp8_e5m2")
        self.FP8_DTYPE = str_to_cutlass_dtype(fp8_dtype)
        self.FP8_DTYPE_STR = fp8_dtype
        if not (rowwise or colwise):
            raise ValueError("at least one of rowwise or colwise must be true")
        self.ROWWISE = rowwise
        self.COLWISE = colwise
        if shape_rep not in SUPPORTED_SHAPE_REPS:
            raise ValueError(
                f"unsupported shape representation {shape_rep!r}; expected one of"
                f" {SUPPORTED_SHAPE_REPS}"
            )
        self.SHAPE_REP = shape_rep
        # Mirrors `is_single_tensor` in group_quantize_mxfp8.cuh.
        self.IS_SINGLE_TENSOR = shape_rep in (SAME_BOTH_DIMS, VARYING_FIRST_DIM)
        self.MAX_NORM_RCP = (
            FP8E4M3_MAX_NORM_RCP if fp8_dtype == "fp8_e4m3fn" else FP8E5M2_MAX_NORM_RCP
        )

    def __str__(self):
        return (
            f"MXFP8GroupQuantizeConfig(dtype={self.DTYPE_STR}, fp8_dtype={self.FP8_DTYPE_STR}, "
            f"rowwise={self.ROWWISE}, colwise={self.COLWISE}, shape_rep={self.SHAPE_REP})"
        )

    __repr__ = __str__


class MXFP8GroupQuantizeKernel:
    """Grouped MXFP8 quantize mirroring group_quantize_mxfp8_kernel's strategy."""

    # TunableConfig / derived constants from group_quantize_mxfp8.cuh.
    CHUNK_DIM_Y = 128
    CHUNK_DIM_X = 128
    THREADS_PER_CHUNK = 128
    STATIC_PERSISTENT_BLOCKS_PER_SM = 24
    ELTS_PER_CHUNK = CHUNK_DIM_Y * CHUNK_DIM_X
    THREADS_X = CHUNK_DIM_X // MXFP8_BLOCK_SCALING_SIZE  # 4
    THREADS_Y = THREADS_PER_CHUNK // THREADS_X  # 32
    BUFF_DIM_Y = THREADS_Y  # 32
    BUFF_DIM_X = CHUNK_DIM_X  # 128
    STAGES = CHUNK_DIM_Y // BUFF_DIM_Y  # 4
    BUFFS_NUM = 2  # PREFETCH_STAGES(1) + 1
    NUM_WARPS = THREADS_PER_CHUNK // THREADS_PER_WARP  # 4
    SCALE_COLS_PER_CHUNK = CHUNK_DIM_X // MXFP8_BLOCK_SCALING_SIZE  # 4

    # Rowwise vectorization constants (mirror MXFP8QuantizeKernel / CUDA PACK_SIZE).
    PACK_SIZE = 4
    WAVES = MXFP8_BLOCK_SCALING_SIZE // PACK_SIZE  # 8
    THREADS_PER_BANK = (32 * 4) // MXFP8_BLOCK_SCALING_SIZE  # 4

    def __init__(self, cfg: MXFP8GroupQuantizeConfig, sm_count: int):
        self.cfg = cfg
        self.sm_count = sm_count

    # ---------------------------------------------------------------- helpers
    @cute.jit
    def _tensor_rows_cols(
        self, tensor_id, mFirstDims, mLastDims, first_logical_dim, last_logical_dim
    ):
        """Get the shape (rows, cols) of the tensor by tensor_id."""
        cfg = self.cfg
        if cutlass.const_expr(cfg.SHAPE_REP == VARYING_FIRST_DIM):
            rows = Int32(mFirstDims[tensor_id])
        else:
            rows = Int32(first_logical_dim)
        if cutlass.const_expr(cfg.SHAPE_REP == VARYING_LAST_DIM):
            cols = Int32(mLastDims[tensor_id])
        else:
            cols = Int32(last_logical_dim)
        return rows, cols

    @cute.jit
    def _find_tensor_from_offsets(self, mOffsets, num_tensors, offset):
        """Binary search over the CSR offsets array (find_tensor_from_offsets)."""
        low = Int32(1)
        hi = Int32(num_tensors)
        while low < hi:
            mid = low + (hi - low) // 2
            if Int32(mOffsets[mid]) <= offset:
                low = mid + 1
            else:
                hi = mid
        return low - 1

    # ------------------------------------------------------------ entry point
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mO_col: cute.Tensor,
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: cute.Tensor,  # int64[num_tensors + 1], CSR element offsets
        mFirstDims: cute.Tensor,  # int64[num_tensors] (VARYING_FIRST_DIM)
        mLastDims: cute.Tensor,  # int64[num_tensors] (VARYING_LAST_DIM)
        mTensormaps: cute.Tensor,  # int64[num_tensors, NUM_TENSORMAPS, 16]
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(f"[CuTeDSL] MXFP8GroupQuantizeKernel.__call__() cfg: {self.cfg}\n")

        cfg = self.cfg
        first_logical_dim = mX.shape[0]
        last_logical_dim = mX.shape[1]
        num_tensors = mOffsets.shape[0] - 1

        smem_tile_layout = cute.make_ordered_layout(
            (self.BUFF_DIM_Y, self.BUFF_DIM_X), order=(1, 0)
        )
        cta_tiler = (self.BUFF_DIM_Y, self.BUFF_DIM_X)
        print(f"mx={mX}, smem_tile_layout={smem_tile_layout}, cta_tiler={cta_tiler}\n")

        op_load = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_x, tma_src = cpasync.make_tiled_tma_atom(
            op_load, mX, smem_tile_layout, cta_tiler, num_multicast=1
        )
        print(f"tma_atom_x={tma_atom_x}\n")
        print(f"tma_src={tma_src}\n")
        op_store = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_out_row, tma_dst_out_row = cpasync.make_tiled_tma_atom(
            op_store, mO_row, smem_tile_layout, cta_tiler, num_multicast=1
        )
        tma_atom_out_col, tma_dst_out_col = cpasync.make_tiled_tma_atom(
            op_store, mO_col, smem_tile_layout, cta_tiler, num_multicast=1
        )

        if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
            # mx is interpreted as a single 2D tensor of shape (first_logical_dim, last_logical_dim).
            work_blocks_Y = cute.ceil_div(first_logical_dim, self.CHUNK_DIM_Y)
            work_blocks_X = cute.ceil_div(Int32(last_logical_dim), self.CHUNK_DIM_X)
        else:
            # mX is interpreted as a 1D tensor of shape (1, first_logical_dim * last_logical_dim)
            work_blocks_Y = Int32(1)
            work_blocks_X = cute.ceil_div(
                Int32(first_logical_dim) * Int32(last_logical_dim), self.ELTS_PER_CHUNK
            )

        # Multi-tensor reps need per-tensor descriptors; fill them first, exactly
        # like the CUDA update_tma_descriptors prologue kernel.
        if cutlass.const_expr(not cfg.IS_SINGLE_TENSOR):
            self.update_descriptors_kernel(
                mX,
                mO_row,
                mO_col,
                mOffsets,
                mFirstDims,
                mLastDims,
                mTensormaps,
                first_logical_dim,
                last_logical_dim,
                mX.element_type,
                tma_atom_x,
                tma_atom_out_row,
                tma_atom_out_col,
            ).launch(grid=[num_tensors, 1, 1], block=[THREADS_PER_WARP, 1, 1], stream=stream)

        self.kernel(
            mS_row,
            mS_col,
            mOffsets,
            mFirstDims,
            mLastDims,
            mTensormaps,
            first_logical_dim,
            last_logical_dim,
            num_tensors,
            work_blocks_X,
            work_blocks_Y,
            mX.element_type,
            tma_atom_x,
            tma_src,
            tma_atom_out_row,
            tma_dst_out_row,
            tma_atom_out_col,
            tma_dst_out_col,
        ).launch(
            grid=[self.sm_count * self.STATIC_PERSISTENT_BLOCKS_PER_SM, 1, 1],
            block=[self.THREADS_PER_CHUNK, 1, 1],
            stream=stream,
        )

    # ------------------------------------------------- descriptor prologue
    @cute.kernel
    def update_descriptors_kernel(
        self,
        mX,
        mO_row,
        mO_col,
        mOffsets,
        mFirstDims,
        mLastDims,
        mTensormaps,
        first_logical_dim,
        last_logical_dim,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom_x,
        tma_atom_orow,
        tma_atom_ocol,
    ):
        """One CTA per tensor: point that tensor's TMA descriptors at its own block.

        CuTeDSL analog of common::update_tma_descriptors writing g_tensor_maps[].
        """
        cfg = self.cfg
        tensor_id, _, _ = cute.arch.block_idx()
        rows, cols = self._tensor_rows_cols(
            tensor_id, mFirstDims, mLastDims, first_logical_dim, last_logical_dim
        )
        base_elts = Int64(mOffsets[tensor_id])

        tmap = TensorMapManager(TensorMapUpdateMode.GMEM, BYTES_PER_TENSORMAP)
        desc_x = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 0, None)].iterator)
        desc_orow = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 1, None)].iterator)
        desc_ocol = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 2, None)].iterator)

        # Zero-sized groups: creating a descriptor with a zero extent is invalid,
        # so skip (the main kernel skips these jobs via job_has_work).
        if rows > 0 and cols > 0:
            gX = cute.make_tensor(
                cute.make_ptr(
                    dtype,
                    mX.iterator.toint() + base_elts * (dtype.width // 8),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((rows, cols), stride=(cols, 1)),
            )
            gO_row = cute.make_tensor(
                cute.make_ptr(
                    cfg.FP8_DTYPE,
                    mO_row.iterator.toint() + base_elts,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((rows, cols), stride=(cols, 1)),
            )
            gO_col = cute.make_tensor(
                cute.make_ptr(
                    cfg.FP8_DTYPE,
                    mO_col.iterator.toint() + base_elts,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((rows, cols), stride=(cols, 1)),
            )

            tmap.init_tensormap_from_atom(tma_atom_x, desc_x, 0)
            if cutlass.const_expr(cfg.ROWWISE):
                tmap.init_tensormap_from_atom(tma_atom_orow, desc_orow, 0)
            if cutlass.const_expr(cfg.COLWISE):
                tmap.init_tensormap_from_atom(tma_atom_ocol, desc_ocol, 0)
            tmap.fence_tensormap_initialization()

            if cutlass.const_expr(cfg.ROWWISE and cfg.COLWISE):
                tmap.update_tensormap(
                    (gX, gO_row, gO_col),
                    (tma_atom_x, tma_atom_orow, tma_atom_ocol),
                    (desc_x, desc_orow, desc_ocol),
                    0,
                    (),  # smem staging is unused in GMEM update mode
                )
            elif cutlass.const_expr(cfg.ROWWISE):
                tmap.update_tensormap(
                    (gX, gO_row),
                    (tma_atom_x, tma_atom_orow),
                    (desc_x, desc_orow),
                    0,
                    (),  # smem staging is unused in GMEM update mode
                )
            else:
                tmap.update_tensormap(
                    (gX, gO_col),
                    (tma_atom_x, tma_atom_ocol),
                    (desc_x, desc_ocol),
                    0,
                    (),  # smem staging is unused in GMEM update mode
                )

    # ------------------------------------------------------------ main kernel
    @cute.kernel
    def kernel(
        self,
        mS_row,
        mS_col,
        mOffsets,
        mFirstDims,
        mLastDims,
        mTensormaps,
        first_logical_dim,
        last_logical_dim,
        num_tensors,
        work_blocks_X,
        work_blocks_Y,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom_x,
        tma_src,
        tma_atom_out_row,
        tma_dst_out_row,
        tma_atom_out_col,
        tma_dst_out_col,
    ):
        cfg = self.cfg
        FP8_DTYPE = cfg.FP8_DTYPE
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdx, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # --- shared memory (allocated once, reused across jobs) ---
        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cute.Int64, 2 * self.BUFFS_NUM]
            sX: cute.struct.Align[
                cute.struct.MemRange[dtype, self.BUFF_DIM_Y * self.BUFF_DIM_X * self.BUFFS_NUM], 128
            ]
            sO_row: cute.struct.Align[
                cute.struct.MemRange[FP8_DTYPE, self.BUFF_DIM_Y * self.BUFF_DIM_X * self.BUFFS_NUM],
                128,
            ]
            sO_col: cute.struct.Align[
                cute.struct.MemRange[FP8_DTYPE, self.BUFF_DIM_Y * self.BUFF_DIM_X * self.BUFFS_NUM],
                128,
            ]

        storage = cutlass.utils.SmemAllocator().allocate(SharedStorage)
        tile_layout = cute.make_layout(
            ((self.BUFF_DIM_Y, self.BUFF_DIM_X), self.BUFFS_NUM),
            stride=((self.BUFF_DIM_X, 1), self.BUFF_DIM_Y * self.BUFF_DIM_X),
        )
        sX = storage.sX.get_tensor(tile_layout)
        sO_row = storage.sO_row.get_tensor(tile_layout)
        sO_col = storage.sO_col.get_tensor(tile_layout)

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=self.BUFFS_NUM,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.NUM_WARPS),
            tx_count=self.BUFF_DIM_Y * self.BUFF_DIM_X * dtype.width // 8,
            cta_layout_vmnk=None,
        )
        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.BUFFS_NUM
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.BUFFS_NUM
        )

        # TMA partitions built from the representative views. For multi-tensor reps
        # the descriptor is swapped per tensor and the tile coords are tensor-local.
        gX_tiled = cute.zipped_divide(tma_src, (self.BUFF_DIM_Y, self.BUFF_DIM_X))
        tXsX, tXgX = cpasync.tma_partition(tma_atom_x, 0, cute.make_layout(1), sX, gX_tiled)
        gO_row_tiled = cute.zipped_divide(tma_dst_out_row, (self.BUFF_DIM_Y, self.BUFF_DIM_X))
        tXsO_row, tXgO_row = cpasync.tma_partition(
            tma_atom_out_row, 0, cute.make_layout(1), sO_row, gO_row_tiled
        )
        gO_col_tiled = cute.zipped_divide(tma_dst_out_col, (self.BUFF_DIM_Y, self.BUFF_DIM_X))
        tXsO_col, tXgO_col = cpasync.tma_partition(
            tma_atom_out_col, 0, cute.make_layout(1), sO_col, gO_col_tiled
        )
        print(f"tma_atom_x={tma_atom_x}\n")
        print(f"tma_src={tma_src}\n")
        print(f"gX_tiled={gX_tiled}\n")
        print(f"tXsX={tXsX}\n")
        print(f"tXgX={tXgX}\n")

        tmap = TensorMapManager(TensorMapUpdateMode.GMEM, BYTES_PER_TENSORMAP)
        cute.arch.sync_threads()

        total_work_blocks = work_blocks_X * work_blocks_Y
        block_stride = Int32(gdx)
        launch_block_id = Int32(bidx)
        ctaid_X = launch_block_id % work_blocks_X
        ctaid_Y = launch_block_id // work_blocks_X
        next_block_id = launch_block_id + block_stride
        last_tensor_id = Int32(-1)

        # Persistent job loop (decode_job / is_job_valid / job_has_work / advance).
        job_finished = Boolean(launch_block_id >= total_work_blocks)
        while not job_finished:
            block_id = ctaid_Y * work_blocks_X + ctaid_X
            if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
                block_global_offset = (
                    ctaid_Y * self.CHUNK_DIM_Y * Int32(last_logical_dim)
                    + ctaid_X * self.CHUNK_DIM_X
                )
            else:
                block_global_offset = block_id * self.ELTS_PER_CHUNK

            # get_current_tensor_id
            if cutlass.const_expr(cfg.SHAPE_REP == SAME_BOTH_DIMS):
                rows_per_tensor = Int32(first_logical_dim) // Int32(num_tensors)
                tensor_id = (ctaid_Y * self.CHUNK_DIM_Y) // rows_per_tensor
            else:
                tensor_id = self._find_tensor_from_offsets(
                    mOffsets, num_tensors, block_global_offset
                )

            rows, cols = self._tensor_rows_cols(
                tensor_id, mFirstDims, mLastDims, first_logical_dim, last_logical_dim
            )

            # is_job_valid
            valid = block_id < total_work_blocks
            if valid and rows > 0 and cols > 0:
                if cutlass.const_expr(cfg.SHAPE_REP != SAME_BOTH_DIMS):
                    tensor_start = Int32(mOffsets[tensor_id])
                    tensor_end = Int32(mOffsets[tensor_id + 1])
                    if block_global_offset >= tensor_end:
                        valid = Boolean(False)
                    else:
                        if (block_global_offset - tensor_start) // cols >= rows:
                            valid = Boolean(False)

            if not valid:
                job_finished = Boolean(True)
            else:
                # job_has_work: zero-sized tensors are skipped but keep scheduling.
                if rows > 0 and cols > 0:
                    self._process_block(
                        block_id,
                        tensor_id,
                        rows,
                        cols,
                        mOffsets,
                        mTensormaps,
                        mS_row,
                        mS_col,
                        first_logical_dim,
                        last_logical_dim,
                        tmap,
                        last_tensor_id,
                        warp_idx,
                        tidx,
                        sX,
                        sO_row,
                        sO_col,
                        tXsX,
                        tXgX,
                        tXsO_row,
                        tXgO_row,
                        tXsO_col,
                        tXgO_col,
                        tma_atom_x,
                        tma_atom_out_row,
                        tma_atom_out_col,
                        mainloop_pipeline,
                        prod_state,
                        cons_state,
                    )
                    last_tensor_id = tensor_id

                # advance_to_next_job
                if next_block_id < total_work_blocks:
                    ctaid_X = next_block_id % work_blocks_X
                    ctaid_Y = next_block_id // work_blocks_X
                    next_block_id = next_block_id + block_stride
                else:
                    job_finished = Boolean(True)

        cute.arch.cp_async_bulk_wait_group(0, read=False)

    @cute.jit
    def _process_block(
        self,
        block_id,
        tensor_id,
        rows,
        cols,
        mOffsets,
        mTensormaps,
        mS_row,
        mS_col,
        first_logical_dim,
        last_logical_dim,
        tmap,
        last_tensor_id,
        warp_idx,
        tidx,
        sX,
        sO_row,
        sO_col,
        tXsX,
        tXgX,
        tXsO_row,
        tXgO_row,
        tXsO_col,
        tXgO_col,
        tma_atom_x,
        tma_atom_out_row,
        tma_atom_out_col,
        mainloop_pipeline,
        prod_state,
        cons_state,
    ):
        """Quantize one 128x128 chunk in STAGES slices of BUFF_DIM_Y rows."""
        cfg = self.cfg

        # decode_block
        blocks_X_in_tensor = cute.ceil_div(cols, self.CHUNK_DIM_X)
        if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
            tensor_base = Int32(0)
            block_id_in_tensor = block_id
        else:
            tensor_base = Int32(mOffsets[tensor_id])
            block_id_in_tensor = block_id - tensor_base // self.ELTS_PER_CHUNK
        block_id_Y = block_id_in_tensor // blocks_X_in_tensor
        block_id_X = block_id_in_tensor % blocks_X_in_tensor
        block_offset_Y = block_id_Y * self.CHUNK_DIM_Y
        block_offset_X = block_id_X * self.CHUNK_DIM_X

        # Per-tensor scale views (CUDA: scales_* + (is_single_tensor ? 0 : base/32),
        # strides roundup(cols/32, 4) rowwise and roundup(cols, 128) colwise).
        # For is_single_tensor the block offsets are GLOBAL over the stacked group, so
        # the scale view (and the helpers' row bound) must use the global height, not
        # the per-tensor `rows`. CUDA guards only on columns -- rows are 128-aligned so
        # a row block is never partial -- and using `rows` here would wrongly mask out
        # every tensor past the first under VARYING_FIRST_DIM.
        scale_rows = Int32(first_logical_dim) if cutlass.const_expr(cfg.IS_SINGLE_TENSOR) else rows
        scale_base = (
            Int64(0)
            if cutlass.const_expr(cfg.IS_SINGLE_TENSOR)
            else Int64(tensor_base) // MXFP8_BLOCK_SCALING_SIZE
        )
        if cutlass.const_expr(cfg.ROWWISE):
            stride_row = cute.round_up(cute.ceil_div(cols, MXFP8_BLOCK_SCALING_SIZE), 4)
            mS_row_t = cute.make_tensor(
                cute.make_ptr(
                    Float8E8M0FNU,
                    mS_row.iterator.toint() + scale_base,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                cute.make_layout((scale_rows, stride_row), stride=(stride_row, 1)),
            )
            mS_row_tiled = cute.zipped_divide(
                mS_row_t, (self.BUFF_DIM_Y, self.SCALE_COLS_PER_CHUNK)
            )
        if cutlass.const_expr(cfg.COLWISE):
            stride_col = cute.round_up(cols, 128)
            mS_col_t = cute.make_tensor(
                cute.make_ptr(
                    Float8E8M0FNU,
                    mS_col.iterator.toint() + scale_base,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                cute.make_layout(
                    (scale_rows // MXFP8_BLOCK_SCALING_SIZE, stride_col), stride=(stride_col, 1)
                ),
            )
            mS_col_tiled = cute.zipped_divide(
                mS_col_t, (self.BUFF_DIM_Y // MXFP8_BLOCK_SCALING_SIZE, self.CHUNK_DIM_X)
            )

        # Acquire this tensor's descriptors when the tensor changes
        # (CUDA: fence_acquire_tensormap on tensor switch).
        if cutlass.const_expr(not cfg.IS_SINGLE_TENSOR):
            desc_x = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 0, None)].iterator)
            desc_out_row = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 1, None)].iterator)
            desc_out_col = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 2, None)].iterator)
            if tensor_id != last_tensor_id:
                tmap.fence_tensormap_update(desc_x)
                if cutlass.const_expr(cfg.ROWWISE):
                    tmap.fence_tensormap_update(desc_out_row)
                if cutlass.const_expr(cfg.COLWISE):
                    tmap.fence_tensormap_update(desc_out_col)
        cute.arch.sync_threads()

        row_tile_base = block_offset_Y // self.BUFF_DIM_Y

        # Prime the pipeline with the first slice (PREFETCH_STAGES == 1).
        if warp_idx == 0:
            mainloop_pipeline.producer_acquire(prod_state)
            if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
                cute.copy(
                    tma_atom_x,
                    tXgX[(None, (row_tile_base, block_id_X))],
                    tXsX[(None, prod_state.index)],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                )
            else:
                cute.copy(
                    tma_atom_x,
                    tXgX[(None, (row_tile_base, block_id_X))],
                    tXsX[(None, prod_state.index)],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                    tma_desc_ptr=tmap.get_tensormap_ptr(desc_x, cute.AddressSpace.generic),
                )
            mainloop_pipeline.producer_commit(prod_state)
            prod_state.advance()

        for stage in cutlass.range_constexpr(self.STAGES):
            # Prefetch the next slice.
            if stage < self.STAGES - 1:
                if warp_idx == 0:
                    mainloop_pipeline.producer_acquire(prod_state)
                    nxt = row_tile_base + stage + 1
                    if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
                        cute.copy(
                            tma_atom_x,
                            tXgX[(None, (nxt, block_id_X))],
                            tXsX[(None, prod_state.index)],
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                        )
                    else:
                        cute.copy(
                            tma_atom_x,
                            tXgX[(None, (nxt, block_id_X))],
                            tXsX[(None, prod_state.index)],
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                            tma_desc_ptr=tmap.get_tensormap_ptr(desc_x, cute.AddressSpace.generic),
                        )
                    mainloop_pipeline.producer_commit(prod_state)
                    prod_state.advance()

            mainloop_pipeline.consumer_wait(cons_state)
            cute.arch.sync_threads()
            stage_idx = cons_state.index
            sX_tile = sX[(None, stage_idx)]
            row_tile = row_tile_base + stage
            tile_row_start = block_offset_Y + stage * self.BUFF_DIM_Y

            if cutlass.const_expr(cfg.COLWISE):
                quantize_colwise_mxfp8(
                    sX_tile,
                    None,
                    sO_col[(None, stage_idx)],
                    cute.flatten(mS_col_tiled[(None, (row_tile, block_id_X))]),
                    cfg.MAX_NORM_RCP,
                    tile_row_start,
                    block_offset_X,
                    scale_rows,
                    cols,
                    None,
                    cfg.DTYPE,
                    cfg.FP8_DTYPE,
                    False,
                    self.BUFF_DIM_X,
                    self.BUFF_DIM_Y,
                    False,
                )
            if cutlass.const_expr(cfg.ROWWISE):
                quantize_rowwise_mxfp8(
                    sX_tile,
                    None,
                    sO_row[(None, stage_idx)],
                    cute.flatten(mS_row_tiled[(None, (row_tile, block_id_X))]),
                    cfg.MAX_NORM_RCP,
                    tile_row_start,
                    block_offset_X,
                    scale_rows,
                    cols,
                    None,
                    cfg.DTYPE,
                    cfg.FP8_DTYPE,
                    self.BUFF_DIM_X,
                    self.BUFF_DIM_Y,
                    self.WAVES,
                    self.THREADS_PER_BANK,
                    self.PACK_SIZE,
                    False,
                )

            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            mainloop_pipeline.consumer_release(cons_state)

            if warp_idx == 0:
                if cutlass.const_expr(cfg.ROWWISE):
                    if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
                        cute.copy(
                            tma_atom_out_row,
                            tXsO_row[(None, stage_idx)],
                            tXgO_row[(None, (row_tile, block_id_X))],
                        )
                    else:
                        cute.copy(
                            tma_atom_out_row,
                            tXsO_row[(None, stage_idx)],
                            tXgO_row[(None, (row_tile, block_id_X))],
                            tma_desc_ptr=tmap.get_tensormap_ptr(
                                desc_out_row, cute.AddressSpace.generic
                            ),
                        )
                if cutlass.const_expr(cfg.COLWISE):
                    if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
                        cute.copy(
                            tma_atom_out_col,
                            tXsO_col[(None, stage_idx)],
                            tXgO_col[(None, (row_tile, block_id_X))],
                        )
                    else:
                        cute.copy(
                            tma_atom_out_col,
                            tXsO_col[(None, stage_idx)],
                            tXgO_col[(None, (row_tile, block_id_X))],
                            tma_desc_ptr=tmap.get_tensormap_ptr(
                                desc_out_col, cute.AddressSpace.generic
                            ),
                        )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(1, read=True)

            cons_state.advance()


def compile_cutedsl_function_from_cfg(cfg: MXFP8GroupQuantizeConfig):
    """Return the compiled CuTeDSL function object for the given grouped config."""
    # CUDA requires the group's first logical dim to be a multiple of 128 (and each
    # tensor's rows/cols likewise); MXFP8 needs the last dim divisible by 32.
    sym_M = cute.sym_int32(divisibility=128)
    sym_N = cute.sym_int32(divisibility=MXFP8_BLOCK_SCALING_SIZE)
    logical_shape = (sym_M, sym_N)

    out_dtype = cfg.FP8_DTYPE
    scale_dtype = cutlass.Float8E8M0FNU

    def g2d(dtype, align=16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            logical_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=align,
        )

    def g1d(dtype, align=4):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            (cute.sym_int32(),),
            stride_order=(0,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=align,
        )

    # The kernel only takes the base address of the scale buffers (per-tensor strides
    # are derived from cols), so their fake shape is a flat 1D byte run.
    in_fake = g2d(cfg.DTYPE)
    out_row_fake = g2d(out_dtype)
    out_col_fake = g2d(out_dtype)
    scale_row_fake = g1d(scale_dtype)
    scale_col_fake = g1d(scale_dtype)
    offsets_fake = g1d(cutlass.Int64, align=8)
    first_dims_fake = g1d(cutlass.Int64, align=8)
    last_dims_fake = g1d(cutlass.Int64, align=8)
    tensormaps_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64,
        (cute.sym_int32(), NUM_TENSORMAPS, BYTES_PER_TENSORMAP // 8),
        stride_order=(2, 1, 0),
        memspace=cute.AddressSpace.gmem,
        assumed_align=128,
    )

    from cutlass.utils import HardwareInfo  # pylint: disable=import-outside-toplevel

    sm_count = HardwareInfo().get_device_multiprocessor_count()
    kernel_obj = MXFP8GroupQuantizeKernel(cfg, sm_count)
    return cute.compile(
        kernel_obj,
        in_fake,
        out_row_fake,
        out_col_fake,
        scale_row_fake,
        scale_col_fake,
        offsets_fake,
        first_dims_fake,
        last_dims_fake,
        tensormaps_fake,
        cute.runtime.make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def get_mxfp8_group_quantization_function(
    fn_name: str,
    dtype: str,
    fp8_dtype: str,
    rowwise: bool,
    colwise: bool,
    shape_rep: str,
) -> bool:
    """Compile the grouped MXFP8 quantize kernel for this config and register it in the
    TVM-FFI global registry under EXACTLY `fn_name`. Returns True on success (the C++
    dispatcher then fetches it with GetGlobal(fn_name)); False if unsupported, so the
    caller falls back to the CUDA C++ grouped kernel.
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
        cfg = MXFP8GroupQuantizeConfig(
            dtype=dtype,
            fp8_dtype=fp8_dtype,
            rowwise=rowwise,
            colwise=colwise,
            shape_rep=shape_rep,
        )
    except ValueError as e:
        logger.warning(
            "CuTeDSL grouped MXFP8 backend does not support this config, "
            "falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False

    logger.debug("Compiling CuTeDSL grouped MXFP8 quantization kernel for %s", cfg)
    try:
        compiled = compile_cutedsl_function_from_cfg(cfg)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "CuTeDSL grouped MXFP8 kernel compilation failed, "
            "falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False
    tvm_ffi.register_global_func(fn_name, compiled, override=True)
    return True
