# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 examples built with :class:`UniversalQuantizer`.

These kernels intentionally cover only plain cast-only configurations with compact,
non-swizzled scale tensors.  Their public call signature matches the existing MXFP8
kernel interface, but amax, noop, activation, and dbias arguments are not consumed.
"""

from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Float32, Float8E8M0FNU, Int32, Uint32

from transformer_engine.common.CuTeDSL.cast.mxfp8.quantize_mxfp8 import (
    MXFP8_BLOCK_SCALING_SIZE,
    THREADS_PER_WARP,
    MXFP8QuantizeConfig,
    MXFP8QuantizeKernelBase,
    quantize_bidimensional_mxfp8_swizzled,
    quantize_rowwise_mxfp8,
)
from transformer_engine.common.CuTeDSL.cast.quantize_universal import (
    UniversalQuantizer,
)
from transformer_engine.common.CuTeDSL.utils import (
    abs_max_x2_bf16,
    abs_max_x2_f16,
    exp2f_rcp,
    fabs_f32,
    is_packed16,
    pack_f32x2,
    x2_hi_to_f32_bf16,
    x2_hi_to_f32_f16,
    x2_lo_to_f32_bf16,
    x2_lo_to_f32_f16,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import (
    cvt_f32_to_fp8e8m0fnu,
    mul_f32x2_cvt_f32x4_to_fp8x4,
    mul_f32x2_cvt_packed16x4_to_fp8x4,
)


def _validate_specialized_config(
    cfg: MXFP8QuantizeConfig,
    *,
    rowwise: bool,
    colwise: bool,
) -> None:
    """Validate the subset represented by the universal prototype."""
    if cfg.ROWWISE != rowwise or cfg.COLWISE != colwise:
        directions = "rowwise+colwise" if colwise else "rowwise-only"
        raise ValueError(f"This kernel requires a {directions} configuration")
    if cfg.WITH_GEMM_SWIZZLED_SCALES:
        raise NotImplementedError(
            "UniversalQuantizer v1 supports only compact, non-swizzled scales"
        )
    if cfg.WITH_AMAX or cfg.WITH_DBIAS or cfg.WITH_DACT or cfg.WITH_ACT:
        raise NotImplementedError(
            "The universal MXFP8 specialized kernels currently support cast-only configurations"
        )
    if not is_packed16(cfg.DTYPE):
        raise NotImplementedError(
            "The specialized universal MXFP8 kernels currently support FP16/BF16 input"
        )
    if cfg.USE_2D_QUANTIZATION:
        raise ValueError("Use MXFP8Quantize2DUniversalKernel for 2D block scaling")


class MXFP8QuantizeSpecializedRowwiseUniversalKernel(UniversalQuantizer, MXFP8QuantizeKernelBase):
    """Rowwise-only MXFP8 quantization expressed through the universal builder."""

    TAG = "MXFP8QuantizeSpecializedRowwiseUniversalKernel"
    _TILE_ROWS = 4
    _TILE_COLS = 1024
    _PACK_SIZE = 4
    _WAVES = MXFP8_BLOCK_SCALING_SIZE // _PACK_SIZE
    _THREADS_PER_BANK = (32 * 4) // MXFP8_BLOCK_SCALING_SIZE
    _THREADS_PER_CTA = 128
    _NUM_STAGES = 2
    _NUM_TILES_X = 1
    _NUM_TILES_Y = 1

    def __init__(self, cfg: MXFP8QuantizeConfig):
        _validate_specialized_config(cfg, rowwise=True, colwise=False)
        self.cfg = cfg
        super().__init__(
            DST="SMEM",
            THREADS_PER_CTA=self._THREADS_PER_CTA,
            TILER=(self._TILE_ROWS, self._TILE_COLS),
            ROWWISE=True,
            COLWISE=False,
            INPUT_DTYPE=cfg.DTYPE,
            OUTPUT_DTYPE=cfg.FP8_DTYPE,
            SCALE_DTYPE=Float8E8M0FNU,
            MX_BLOCK_SIZE=MXFP8_BLOCK_SCALING_SIZE,
            STASH_SCALE_TO_SMEM=True,
            NUM_TILES_X=self._NUM_TILES_X,
            NUM_TILES_Y=self._NUM_TILES_Y,
            PIPELINE_DEPTH=self._NUM_STAGES,
            USE_TMA=True,
        )

    # Unused parameters are required by UniversalQuantizer's callback ABI.
    # pylint: disable=unused-argument
    @cute.jit
    def _TILE_QUANTIZER_FUNC(
        self,
        sX: cute.Tensor,
        sO_row: cute.Tensor,
        sS_row: cute.Tensor,
        sO_col: Optional[cute.Tensor],
        sS_col: Optional[cute.Tensor],
        scratch,
        tidx: Int32,
        tile_row: Int32,
        tile_col: Int32,
        rows: Int32,
        cols: Int32,
    ):
        """Quantize one input tile rowwise."""
        # A stage slice retains the tile as one hierarchical mode, while the
        # existing MXFP8 routine indexes scales as a flat rank-2 tensor.
        sS_row_2d = cute.make_tensor(
            sS_row.iterator,
            cute.make_layout(
                (self._TILE_ROWS, self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE),
                stride=(self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE, 1),
            ),
        )
        quantize_rowwise_mxfp8(
            sX,
            None,
            sO_row,
            sS_row_2d,
            self.cfg.MAX_NORM_RCP,
            tile_row,
            tile_col,
            rows,
            cols,
            ACTIVATION=None,
            DTYPE=self.cfg.DTYPE,
            FP8_DTYPE=self.cfg.FP8_DTYPE,
            TILE_X=self._TILE_COLS,
            TILE_Y=self._TILE_ROWS,
            WAVES=self._WAVES,
            THREADS_PER_BANK=self._THREADS_PER_BANK,
            PACK_SIZE=self._PACK_SIZE,
            SKIP_INPUT_MASKING=True,
            SKIP_SCALE_BOUNDS=True,
        )

    # pylint: enable=unused-argument


class MXFP8QuantizeSpecializedBidimensionalUniversalKernel(
    UniversalQuantizer, MXFP8QuantizeKernelBase
):
    """Rowwise+colwise MXFP8 quantization expressed through the universal builder."""

    TAG = "MXFP8QuantizeSpecializedBidimensionalUniversalKernel"
    _WARPS_PER_CTA = 2
    _THREADS_PER_CTA = _WARPS_PER_CTA * THREADS_PER_WARP
    _TILE_ROWS = MXFP8_BLOCK_SCALING_SIZE
    _TILE_COLS = MXFP8_BLOCK_SCALING_SIZE * _WARPS_PER_CTA
    _NUM_STAGES = 2
    _NUM_TILES_X = 4
    _NUM_TILES_Y = 1

    def __init__(self, cfg: MXFP8QuantizeConfig):
        _validate_specialized_config(cfg, rowwise=True, colwise=True)
        self.cfg = cfg
        super().__init__(
            DST="SMEM",
            THREADS_PER_CTA=self._THREADS_PER_CTA,
            TILER=(self._TILE_ROWS, self._TILE_COLS),
            ROWWISE=True,
            COLWISE=True,
            INPUT_DTYPE=cfg.DTYPE,
            OUTPUT_DTYPE=cfg.FP8_DTYPE,
            SCALE_DTYPE=Float8E8M0FNU,
            MX_BLOCK_SIZE=MXFP8_BLOCK_SCALING_SIZE,
            STASH_SCALE_TO_SMEM=True,
            EXTRA_SMEM_FACTORY=self.allocate_scratch,
            NUM_TILES_X=self._NUM_TILES_X,
            NUM_TILES_Y=self._NUM_TILES_Y,
            PIPELINE_DEPTH=self._NUM_STAGES,
            USE_TMA=True,
        )

    @staticmethod
    def allocate_scratch(
        smem: cutlass.utils.SmemAllocator,
        tiler: tuple[int, int],
        threads_per_cta: int,
        pipeline_depth: int,
    ):
        """Allocate one 32-element FP32 column-reduction buffer per warp."""
        del tiler, pipeline_depth
        warps_per_cta = threads_per_cta // THREADS_PER_WARP

        @cute.struct
        class ScratchStorage:
            """Callback-owned per-warp column-reduction storage."""

            col_reduce: cute.struct.Align[
                cute.struct.MemRange[Float32, THREADS_PER_WARP * warps_per_cta],
                16,
            ]

        storage = smem.allocate(ScratchStorage)
        return storage.col_reduce.get_tensor(
            cute.make_layout(
                (THREADS_PER_WARP, warps_per_cta),
                stride=(1, THREADS_PER_WARP),
            )
        )

    # Unused parameters are required by UniversalQuantizer's callback ABI.
    # pylint: disable=unused-argument
    @cute.jit
    def _TILE_QUANTIZER_FUNC(
        self,
        sX: cute.Tensor,
        sO_row: cute.Tensor,
        sS_row: cute.Tensor,
        sO_col: cute.Tensor,
        sS_col: cute.Tensor,
        scratch: cute.Tensor,
        tidx: Int32,
        tile_row: Int32,
        tile_col: Int32,
        rows: Int32,
        cols: Int32,
    ):
        """Quantize one input tile rowwise and columnwise."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        quantize_bidimensional_mxfp8_swizzled(
            sX,
            sO_row,
            sO_col,
            sS_row,
            sS_col,
            scratch[None, warp_idx],
            self._WARPS_PER_CTA,
            self.cfg.MAX_NORM_RCP,
            self.cfg.DTYPE,
            self.cfg.FP8_DTYPE,
        )

    # pylint: enable=unused-argument


@cute.autotune(
    configs=[
        _NUM_TILES_X: [1, 2, 4],
    ]
)
class MXFP8Quantize2DUniversalKernel(UniversalQuantizer, MXFP8QuantizeKernelBase):
    """Cast-only MXFP8 quantization with one scale per 32x32 input block."""

    TAG = "MXFP8Quantize2DUniversalKernel"
    _WARPS_PER_CTA = 2
    _THREADS_PER_CTA = _WARPS_PER_CTA * THREADS_PER_WARP
    _TILE_ROWS = MXFP8_BLOCK_SCALING_SIZE
    _TILE_COLS = MXFP8_BLOCK_SCALING_SIZE * _WARPS_PER_CTA
    _NUM_STAGES = 2
    _NUM_TILES_X = 4
    _NUM_TILES_Y = 1

    def __init__(self, cfg: MXFP8QuantizeConfig):
        if not cfg.USE_2D_QUANTIZATION:
            raise ValueError("MXFP8Quantize2DUniversalKernel requires 2D block scaling")
        if cfg.WITH_GEMM_SWIZZLED_SCALES:
            raise NotImplementedError(
                "UniversalQuantizer v1 supports only compact, non-swizzled scales"
            )
        if cfg.WITH_AMAX or cfg.WITH_DBIAS or cfg.WITH_DACT or cfg.WITH_ACT:
            raise NotImplementedError(
                "The universal MXFP8 2D kernel currently supports cast-only configurations"
            )

        self.cfg = cfg
        super().__init__(
            DST="SMEM",
            THREADS_PER_CTA=self._THREADS_PER_CTA,
            TILER=(self._TILE_ROWS, self._TILE_COLS),
            ROWWISE=cfg.ROWWISE,
            COLWISE=cfg.COLWISE,
            INPUT_DTYPE=cfg.DTYPE,
            OUTPUT_DTYPE=cfg.FP8_DTYPE,
            SCALE_DTYPE=Float8E8M0FNU,
            MX_BLOCK_SIZE=MXFP8_BLOCK_SCALING_SIZE,
            STASH_SCALE_TO_SMEM=True,
            NUM_TILES_X=self._NUM_TILES_X,
            NUM_TILES_Y=self._NUM_TILES_Y,
            PIPELINE_DEPTH=self._NUM_STAGES,
            USE_TMA=True,
        )

    # Tile coordinates and extents are unnecessary because TMA zero-fills partial
    # blocks and the builder predicates the corresponding scale flushes.
    # pylint: disable=unused-argument
    @cute.jit
    def _TILE_QUANTIZER_FUNC(
        self,
        sX: cute.Tensor,
        sO_row: Optional[cute.Tensor],
        sS_row: Optional[cute.Tensor],
        sO_col: Optional[cute.Tensor],
        sS_col: Optional[cute.Tensor],
        scratch,
        tidx: Int32,
        tile_row: Int32,
        tile_col: Int32,
        rows: Int32,
        cols: Int32,
    ):
        """Quantize one input tile with 2D block scaling."""
        # A warp owns one 32x32 block: lane selects the row and warp selects
        # which of the two horizontal blocks in the CTA tile it processes.
        _, tv_data = cute.make_layout_tv(
            thr_layout=cute.make_layout(((MXFP8_BLOCK_SCALING_SIZE, 1), self._WARPS_PER_CTA)),
            val_layout=cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE)),
        )
        tXsX = cute.composition(sX, tv_data)[tidx, None]

        if cutlass.const_expr(is_packed16(self.cfg.DTYPE)):
            rX = cute.make_rmem_tensor(MXFP8_BLOCK_SCALING_SIZE, self.cfg.DTYPE)
            cute.autovec_copy(tXsX, rX)
            rX_2x = cute.make_tensor(
                cute.recast_ptr(rX.iterator, dtype=Int32),
                cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 2,), stride=(1,)),
            )
            abs_max_x2 = abs_max_x2_f16 if self.cfg.DTYPE is cutlass.Float16 else abs_max_x2_bf16
            x2_lo_to_f32 = (
                x2_lo_to_f32_f16 if self.cfg.DTYPE is cutlass.Float16 else x2_lo_to_f32_bf16
            )
            x2_hi_to_f32 = (
                x2_hi_to_f32_f16 if self.cfg.DTYPE is cutlass.Float16 else x2_hi_to_f32_bf16
            )
            row_amax_2x = rX_2x[0]
            for i in cutlass.range_constexpr(1, MXFP8_BLOCK_SCALING_SIZE // 2):
                row_amax_2x = abs_max_x2(row_amax_2x, rX_2x[i])
            row_amax = cute.arch.fmax(
                fabs_f32(x2_lo_to_f32(row_amax_2x)),
                fabs_f32(x2_hi_to_f32(row_amax_2x)),
            )
        else:
            rX = cute.make_rmem_tensor(MXFP8_BLOCK_SCALING_SIZE, Float32)
            cute.autovec_copy(tXsX, rX)
            row_amax = Float32(0.0)
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
                row_amax = cute.arch.fmax(row_amax, fabs_f32(rX[i]))

        # Every lane receives the same 32x32-block maximum and therefore the
        # same exponent. Repeating it through both metadata views matches the
        # CUDA kIs2DBlockScaling representation.
        block_amax = cute.arch.warp_redux_sync(row_amax, kind="fmax")
        biased_exp = cvt_f32_to_fp8e8m0fnu(block_amax * self.cfg.MAX_NORM_RCP)

        if cutlass.const_expr(self.cfg.ROWWISE):
            _, tv_scale_row = cute.make_layout_tv(
                thr_layout=cute.make_layout(((MXFP8_BLOCK_SCALING_SIZE, 1), self._WARPS_PER_CTA)),
                val_layout=cute.make_layout((1, 1)),
            )
            cute.composition(sS_row, tv_scale_row)[tidx, None][0] = biased_exp
        if cutlass.const_expr(self.cfg.COLWISE):
            _, tv_scale_col = cute.make_layout_tv(
                thr_layout=cute.make_layout((1, (MXFP8_BLOCK_SCALING_SIZE, self._WARPS_PER_CTA))),
                val_layout=cute.make_layout((1, 1)),
            )
            cute.composition(sS_col, tv_scale_col)[tidx, None][0] = biased_exp

        inverse_scale = exp2f_rcp(biased_exp)
        scale_2x = pack_f32x2(inverse_scale, inverse_scale)
        rO = cute.make_rmem_tensor(MXFP8_BLOCK_SCALING_SIZE, self.cfg.FP8_DTYPE)
        rO_u32 = cute.make_tensor(
            cute.recast_ptr(rO.iterator, dtype=Uint32),
            cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 4,), stride=(1,)),
        )
        if cutlass.const_expr(is_packed16(self.cfg.DTYPE)):
            mul_cvt4 = mul_f32x2_cvt_packed16x4_to_fp8x4(self.cfg.DTYPE, self.cfg.FP8_DTYPE)
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE // 4):
                rO_u32[i] = mul_cvt4(
                    rX_2x[2 * i],
                    rX_2x[2 * i + 1],
                    scale_2x,
                )
        else:
            mul_cvt4 = mul_f32x2_cvt_f32x4_to_fp8x4(self.cfg.FP8_DTYPE)
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE // 4):
                offset = 4 * i
                rO_u32[i] = mul_cvt4(
                    rX[offset],
                    rX[offset + 1],
                    rX[offset + 2],
                    rX[offset + 3],
                    scale_2x,
                )

        if cutlass.const_expr(self.cfg.ROWWISE):
            cute.autovec_copy(rO, cute.composition(sO_row, tv_data)[tidx, None])
        if cutlass.const_expr(self.cfg.COLWISE):
            cute.autovec_copy(rO, cute.composition(sO_col, tv_data)[tidx, None])

    # pylint: enable=unused-argument
