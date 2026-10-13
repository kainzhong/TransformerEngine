# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped GMEM-to-RMEM MXFP8 cast for BF16 input.

Each CTA serves one group member (blockIdx.y), reading the member's rows, columns and element
offset directly -- no search, and, since the tiles are loaded with plain global loads rather
than TMA, no per-member descriptors and no descriptor-update prologue kernel. A member is
viewed as its own (rows, cols) tensor whose scales start at offset / 32, packed or
GEMM-swizzled, and its 32 x 128 tiles are quantized by the register-resident tile function of
the single-tensor bidimensional kernel (quantize_bidimensional_mxfp8_g2r), which runs the
rowwise and/or colwise pass the config asks for.

Requires every member's rows and cols to be multiples of 128 (both grouped kernels require
this for VARYING_LAST_DIM / VARYING_BOTH_DIMS; for the single-tensor representations the
dispatcher checks cols), and fewer than 2^31 elements per member.
"""

from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Int32, Int64
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.utils import create_l2_policy
from .quantize_mxfp8_common import (
    MXFP8_BLOCK_SCALING_SIZE,
    MXFP8QuantizeConfig,
    derive_swizzled_scale_layout,
)
from .quantize_mxfp8_g2r_bidimensional import quantize_bidimensional_mxfp8_g2r
from .group_quantize_mxfp8_common import (
    MXFP8GroupQuantizeConfig,
    SAME_BOTH_DIMS,
    VARYING_FIRST_DIM,
    VARYING_LAST_DIM,
)


class MXFP8GroupQuantizeG2RKernel:
    """Grid of (CTAs per member, num_tensors); each CTA grid-strides its member's tiles."""

    _CTA_THREADS_Y = 8
    # Eight lanes of 16 BF16 values (one 256-bit load each) per tile row: 1-8% faster than
    # sixteen lanes of 8 values (128-bit loads) on GB200.
    _CTA_THREADS_X = 8
    _ELEMENTS_X_PER_THREAD = 16
    _ELEMENTS_Y_PER_THREAD = MXFP8_BLOCK_SCALING_SIZE // _CTA_THREADS_Y  # 4
    TILE_ROWS = _CTA_THREADS_Y * _ELEMENTS_Y_PER_THREAD  # 32
    TILE_COLS = _CTA_THREADS_X * _ELEMENTS_X_PER_THREAD  # 128
    THREADS_PER_CTA = _CTA_THREADS_Y * _CTA_THREADS_X  # 64
    # CTAs per SM the grid is sized for, split evenly across the members. Far more than are
    # resident: on GB200 the loop's latency is better hidden by more CTAs in flight than by
    # each CTA walking more tiles (16 -> 256 per SM took colwise from ~107 to ~92 us at
    # 32768x7168).
    WORKERS_PER_SM = 256

    def __init__(self, cfg: MXFP8GroupQuantizeConfig, sm_count: int) -> None:
        assert self.supports(cfg), f"unsupported config {cfg}"
        self.cfg = cfg
        self.SM_COUNT = sm_count
        self.ROWS_VARY = cfg.SHAPE_REP not in (SAME_BOTH_DIMS, VARYING_LAST_DIM)
        self.COLS_VARY = cfg.SHAPE_REP not in (SAME_BOTH_DIMS, VARYING_FIRST_DIM)
        self.SAME_BOTH_DIMS = cfg.SHAPE_REP == SAME_BOTH_DIMS
        # The tile function is shared with the single-tensor bidimensional kernel and takes
        # its config type.
        self.tile_cfg = MXFP8QuantizeConfig(
            dtype=cfg.DTYPE_STR,
            fp8_dtype=cfg.FP8_DTYPE_STR,
            rowwise=cfg.ROWWISE,
            colwise=cfg.COLWISE,
            with_gemm_swizzled_scales=cfg.WITH_GEMM_SWIZZLED_SCALES,
            with_amax=False,
            activation="none",
        )

    @staticmethod
    def supports(cfg: MXFP8GroupQuantizeConfig) -> bool:
        """BF16 plain casts: no dbias and no fused (d)activation."""
        plain_cast = not (cfg.WITH_DBIAS or cfg.WITH_ACT or cfg.WITH_DACT)
        return plain_cast and cfg.DTYPE is cutlass.BFloat16

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        mLastDims: Optional[cute.Tensor],
        num_tensors: Int32,
        stream: CUstream,
    ) -> None:
        # Spread the persistent worker budget across the members, but never launch more CTAs
        # per member than the average member has tiles.
        num_elems = Int64(mX.shape[0]) * Int64(mX.shape[1])
        tile_elems = self.TILE_ROWS * self.TILE_COLS
        tiles = (num_elems + tile_elems - 1) // tile_elems
        avg_tiles = (tiles + Int64(num_tensors) - 1) // Int64(num_tensors)
        budget = Int64(self.SM_COUNT * self.WORKERS_PER_SM) // Int64(num_tensors)
        ctas_per_member = cutlass.max(Int64(1), cutlass.min(budget, avg_tiles))
        self.kernel(
            mX,
            mO_row,
            mO_col,
            mS_row,
            mS_col,
            mOffsets,
            mFirstDims,
            mLastDims,
            Int64(mX.shape[0]),
            Int64(mX.shape[1]),
            num_tensors,
        ).launch(
            grid=[Int32(ctas_per_member), num_tensors, 1],
            block=[self.THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        mLastDims: Optional[cute.Tensor],
        first_logical_dim: Int64,
        last_logical_dim: Int64,
        num_tensors: Int32,
    ):
        cfg = self.cfg
        bidx, member_id, _ = cute.arch.block_idx()
        gdx, _, _ = cute.arch.grid_dim()

        # This member's extents and element offset in the flat payload.
        rows = Int64(0)
        cols = Int64(last_logical_dim)
        base = Int64(0)
        if cutlass.const_expr(self.SAME_BOTH_DIMS):
            rows = first_logical_dim // Int64(num_tensors)
            base = Int64(member_id) * rows * cols
        else:
            base = Int64(mOffsets[member_id])
            if cutlass.const_expr(self.ROWS_VARY):
                rows = Int64(mFirstDims[member_id])
            else:
                rows = first_logical_dim
            if cutlass.const_expr(self.COLS_VARY):
                cols = Int64(mLastDims[member_id])

        # Member-local extents fit Int32; the 128-divisibility lets the vectorized copies
        # prove their alignment. Members start on 128 x 128 element boundaries and the
        # dispatcher checks the payload is 32-byte aligned, as the 256-bit loads need.
        M = cute.assume(Int32(rows), divby=128)
        N = cute.assume(Int32(cols), divby=128)
        scale_base = base // MXFP8_BLOCK_SCALING_SIZE
        data_layout = cute.make_layout((M, N), stride=(N, 1))
        gX = cute.make_tensor((mX.iterator + base).align(32), data_layout)
        gO_row = gS_row = gO_col = gS_col = None
        if cutlass.const_expr(cfg.ROWWISE):
            gO_row = cute.make_tensor((mO_row.iterator + base).align(16), data_layout)
            gS_row = cute.make_tensor(
                mS_row.iterator + scale_base, cute.make_layout((M, N // 32), stride=(N // 32, 1))
            )
        if cutlass.const_expr(cfg.COLWISE):
            gO_col = cute.make_tensor((mO_col.iterator + base).align(16), data_layout)
            # Member scale offsets are multiples of 512 (members start on 128 x 128 element
            # boundaries) and the dispatcher checks the buffer is 16-byte aligned.
            gS_col = cute.make_tensor(
                (mS_col.iterator + scale_base).align(16),
                cute.make_layout((M // 32, N), stride=(N, 1)),
            )
        if cutlass.const_expr(cfg.WITH_GEMM_SWIZZLED_SCALES):
            gS_row, gS_col = derive_swizzled_scale_layout(
                M, N, cfg.ROWWISE, cfg.COLWISE, gS_row, gS_col
            )

        cache_policy = create_l2_policy(True)
        tiles_X = N // self.TILE_COLS
        num_tiles = (M // self.TILE_ROWS) * tiles_X
        # Grid-stride over this member's tiles (column tiles fastest, as the single-tensor
        # kernels launch them). An empty member has no tiles.
        tile = Int32(bidx)
        while tile < num_tiles:
            coord = (tile // tiles_X, tile % tiles_X)
            self._quantize_tile(gX, gO_row, gS_row, gO_col, gS_col, coord, cache_policy)
            tile += Int32(gdx)

    @cute.jit
    def _quantize_tile(self, gX, gO_row, gS_row, gO_col, gS_col, coord, cache_policy):
        """Quantize the (TILE_ROWS, TILE_COLS) tile at `coord` of one member."""
        cfg = self.cfg
        TILER = (self.TILE_ROWS, self.TILE_COLS)
        ROW_SCALE_TILER = (self.TILE_ROWS, self.TILE_COLS // MXFP8_BLOCK_SCALING_SIZE)
        COL_SCALE_TILER = (self.TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE, self.TILE_COLS)
        gO_row_tile = gS_row_tile = gO_col_tile = gS_col_tile = None
        if cutlass.const_expr(cfg.ROWWISE):
            gO_row_tile = cute.local_tile(gO_row, TILER, coord)
            gS_row_tile = cute.local_tile(gS_row, ROW_SCALE_TILER, coord)
        if cutlass.const_expr(cfg.COLWISE):
            gO_col_tile = cute.local_tile(gO_col, TILER, coord)
            gS_col_tile = cute.local_tile(gS_col, COL_SCALE_TILER, coord)
        quantize_bidimensional_mxfp8_g2r(
            cute.local_tile(gX, TILER, coord),
            gO_row_tile,
            gS_row_tile,
            gO_col_tile,
            gS_col_tile,
            self.tile_cfg,
            CTA_THREADS_Y=self._CTA_THREADS_Y,
            CTA_THREADS_X=self._CTA_THREADS_X,
            ELEMENTS_Y_PER_THREAD=self._ELEMENTS_Y_PER_THREAD,
            ELEMENTS_X_PER_THREAD=self._ELEMENTS_X_PER_THREAD,
            cache_policy=cache_policy,
        )
