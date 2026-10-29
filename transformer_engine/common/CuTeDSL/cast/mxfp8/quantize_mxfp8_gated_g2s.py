# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Gated-activation MXFP8 casts in CuTeDSL, ported from gated_mxfp8.cuh.

The kernel reuses MXFP8QuantizeGeneralG2SKernel's TMA pipeline and its plain-cast SMEM tile
quantizer. What it adds is an elementwise pass that runs on each staged tile before it is
quantized: it combines the act and gate halves of the input (and the incoming gradient, in
backward) and writes the result back into the act (and gate) input buffer in the input dtype.
That is exactly the truncation the CUDA kernel applies before computing block amaxes, so the
plain cast that follows reproduces its output.

Gated input is [M, 2N] with [act | gate] along the last dim. Every operand is addressed through
(M, N) views (stride 2N, column offset 0 or N), mirroring the CUDA kernel's tensor maps:
  forward : out[M, N] = act(x[:, :N]) * gate(x[:, N:])
  backward: out[M, 2N] = [d/d act | d/d gate] given grad[M, N]
"""

from typing import Optional

import cutlass
from cutlass import cute
from cutlass import pipeline
from cutlass import Boolean, Float32, Int32, Int64
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.utils import (
    fabs_f32,
    is_packed16,
    truncate_f32_bf16,
    truncate_f32_f16,
)
from transformer_engine.common.CuTeDSL.activations import (
    act_gelu,
    act_qgelu,
    act_relu,
    act_srelu,
    act_silu,
    act_sigmoid,
    act_clamped_silu,
    act_situ_gate,
    dact_dgelu,
    dact_dqgelu,
    dact_drelu,
    dact_dsrelu,
    dact_dsigmoid,
    dact_dsitu_gate,
    situ_up,
    dsitu_up,
    sigmoidf,
)
from .quantize_mxfp8_common import (
    CUTEDSL_DEBUG_LOGGING,
    MXFP8QuantizeConfig,
    MXFP8_BLOCK_SCALING_SIZE,
    derive_swizzled_scale_layout,
)
from .quantize_mxfp8_g2s_general import (
    MXFP8QuantizeGeneralG2SKernel,
    quantize_mxfp8_smem_tile_general,
)

# Gated activations of gated_mxfp8.cuh, as (ActOP, DActOP) ports of the math.h functions the
# CUDA dispatcher instantiates them with. swiglu and clamped_swiglu backward, and situglu in
# both directions, do not go through this table: like the CUDA kernel they take a special path.
GATED_ACTIVATIONS = {
    "geglu": (act_gelu, dact_dgelu),
    "qgeglu": (act_qgelu, dact_dqgelu),
    "reglu": (act_relu, dact_drelu),
    "sreglu": (act_srelu, dact_dsrelu),
    "glu": (act_sigmoid, dact_dsigmoid),
    "swiglu": (act_silu, None),
    "clamped_swiglu": (None, None),
    "situglu": (None, None),
}


@cute.jit
def _select_f32(pred: Boolean, if_true: Float32, if_false: Float32) -> Float32:
    result = if_false
    if pred:
        result = if_true
    return result


@cute.jit
def _gated_element(
    act_elt: Float32,
    gate_elt: Float32,
    grad_elt: Optional[Float32],
    limit: Float32,
    alpha: Float32,
    glu_linear_offset: Float32,
    beta1: Float32,
    beta2: Float32,
    ACTIVATION: cutlass.Constexpr[str],
    IS_BWD: cutlass.Constexpr[bool],
):
    """One element of the gated op in FP32; returns (act output, gate output or None).

    Mirrors the elementwise math of gated_mxfp8.cuh's columnwise pass (forward and backward),
    operation by operation.
    """
    out_gate = None
    # Transform the gate. Gating is ideally the identity, with derivative 1.
    gate = gate_elt
    dgate = None
    if cutlass.const_expr(ACTIVATION == "clamped_swiglu"):
        # |g| > limit is gated_mxfp8.cuh's (g > limit || g < -limit), NaN included
        dgate = _select_f32(fabs_f32(gate_elt) > limit, Float32(0.0), Float32(1.0))
        gate = cute.arch.fmin(cute.arch.fmax(-limit, gate_elt), limit) + glu_linear_offset
    elif cutlass.const_expr(ACTIVATION == "situglu"):
        dgate = dsitu_up(gate_elt, beta2)
        gate = situ_up(gate_elt, beta2)

    if cutlass.const_expr(not IS_BWD):
        if cutlass.const_expr(ACTIVATION == "clamped_swiglu"):
            act = act_clamped_silu(act_elt, limit, alpha)
        elif cutlass.const_expr(ACTIVATION == "situglu"):
            act = act_situ_gate(act_elt, beta1)
        else:
            act = GATED_ACTIVATIONS[ACTIVATION][0](act_elt)
        out_act = act * gate
    else:
        if cutlass.const_expr(ACTIVATION == "clamped_swiglu"):
            x = cute.arch.fmin(act_elt, limit)
            s = sigmoidf(alpha * x)
            act_x = x * s
            dact_x = _select_f32(
                act_elt <= limit, s + s * (Float32(1.0) - s) * alpha * x, Float32(0.0)
            )
        elif cutlass.const_expr(ACTIVATION == "situglu"):
            act_x = act_situ_gate(act_elt, beta1)
            dact_x = dact_dsitu_gate(act_elt, beta1)
        elif cutlass.const_expr(ACTIVATION == "swiglu"):
            s = sigmoidf(act_elt)
            act_x = act_elt * s
            dact_x = act_elt * s * (Float32(1.0) - s) + s
        else:
            act_fn, dact_fn = GATED_ACTIVATIONS[ACTIVATION]
            act_x = act_fn(act_elt)
            dact_x = dact_fn(act_elt)
        out_act = dact_x * grad_elt * gate
        if cutlass.const_expr(dgate is None):
            out_gate = act_x * grad_elt
        else:
            out_gate = dgate * act_x * grad_elt
    return out_act, out_gate


@cute.jit
def gated_elementwise_smem_tile(
    sAct_tile: cute.Tensor,  # (TILE_Y, TILE_X) act half of the input; receives the act output
    sGate_tile: cute.Tensor,  # (TILE_Y, TILE_X) gate half; receives the gate output (bwd only)
    sGrad_tile: Optional[cute.Tensor],  # (TILE_Y, TILE_X) incoming gradient (bwd only)
    limit: Float32,
    alpha: Float32,
    glu_linear_offset: Float32,
    beta1: Float32,
    beta2: Float32,
    ACTIVATION: cutlass.Constexpr[str],
    IS_BWD: cutlass.Constexpr[bool],
    DTYPE: cutlass.Constexpr[type],
    TILE_X: cutlass.Constexpr[int],
):
    """Compute the gated op on one staged tile, in place, rounded to the input dtype.

    Thread tidx owns column tidx of the tile, which is also the element set it owns in the
    colwise quantization pass, so a colwise-only consumer needs no barrier after this.
    """
    tidx, _, _ = cute.arch.thread_idx()

    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((1, TILE_X), stride=(TILE_X, 1)),
        val_layout=cute.make_layout((MXFP8_BLOCK_SCALING_SIZE, 1), stride=(1, 1)),
    )
    sAct_thread = cute.composition(sAct_tile, tv_layout)[tidx, None]
    sGate_thread = cute.composition(sGate_tile, tv_layout)[tidx, None]
    if cutlass.const_expr(IS_BWD):
        sGrad_thread = cute.composition(sGrad_tile, tv_layout)[tidx, None]
    if cutlass.const_expr(is_packed16(DTYPE)):
        truncate_f32 = truncate_f32_f16 if DTYPE is cutlass.Float16 else truncate_f32_bf16

    for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
        grad_elt = None
        if cutlass.const_expr(IS_BWD):
            grad_elt = Float32(sGrad_thread[i])
        out_act, out_gate = _gated_element(
            Float32(sAct_thread[i]),
            Float32(sGate_thread[i]),
            grad_elt,
            limit,
            alpha,
            glu_linear_offset,
            beta1,
            beta2,
            ACTIVATION=ACTIVATION,
            IS_BWD=IS_BWD,
        )
        # Numerical truncation: round to the input dtype, like the CUDA kernels do before
        # taking the amax. The cast below is then exact.
        if cutlass.const_expr(is_packed16(DTYPE)):
            out_act = truncate_f32(out_act)
            if cutlass.const_expr(IS_BWD):
                out_gate = truncate_f32(out_gate)
        sAct_thread[i] = DTYPE(out_act)
        if cutlass.const_expr(IS_BWD):
            sGate_thread[i] = DTYPE(out_gate)


@cute.jit
def quantize_gated_mxfp8_smem_tile(
    sAct_tile: cute.Tensor,
    sGate_tile: cute.Tensor,
    sGrad_tile: Optional[cute.Tensor],
    sO_row_act_tile: Optional[cute.Tensor],
    mS_row_act_stage: Optional[cute.Tensor],
    sO_row_gate_tile: Optional[cute.Tensor],  # bwd only
    mS_row_gate_stage: Optional[cute.Tensor],  # bwd only
    sO_col_act_tile: Optional[cute.Tensor],
    mS_col_act_stage: Optional[cute.Tensor],
    sO_col_gate_tile: Optional[cute.Tensor],  # bwd only
    mS_col_gate_stage: Optional[cute.Tensor],  # bwd only
    tile_row_start: Int32,
    tile_col_start: Int32,
    M: Int32,
    N: Int32,
    limit: Float32,
    alpha: Float32,
    glu_linear_offset: Float32,
    beta1: Float32,
    beta2: Float32,
    ACTIVATION: cutlass.Constexpr[str],
    IS_BWD: cutlass.Constexpr[bool],
    ROWWISE: cutlass.Constexpr[bool],
    COLWISE: cutlass.Constexpr[bool],
    MAX_NORM_RCP: cutlass.Constexpr[float],
    DTYPE: cutlass.Constexpr[type],
    FP8_DTYPE: cutlass.Constexpr[type],
    WITH_GEMM_SWIZZLED_SCALES: cutlass.Constexpr[bool],
    TILE_X: cutlass.Constexpr[int],
    TILE_Y: cutlass.Constexpr[int],
):
    """Apply the gated op to one staged tile, then MXFP8-quantize the result(s) as a plain cast."""
    gated_elementwise_smem_tile(
        sAct_tile,
        sGate_tile,
        sGrad_tile,
        limit,
        alpha,
        glu_linear_offset,
        beta1,
        beta2,
        ACTIVATION=ACTIVATION,
        IS_BWD=IS_BWD,
        DTYPE=DTYPE,
        TILE_X=TILE_X,
    )
    # The rowwise pass reads elements other threads wrote.
    if cutlass.const_expr(ROWWISE):
        cute.arch.sync_threads()

    # The act output and, backward, the gate output. Both share this tile's coordinates; only
    # their output buffers and scale views differ.
    halves = [(sAct_tile, sO_row_act_tile, mS_row_act_stage, sO_col_act_tile, mS_col_act_stage)]
    if cutlass.const_expr(IS_BWD):
        halves.append(
            (sGate_tile, sO_row_gate_tile, mS_row_gate_stage, sO_col_gate_tile, mS_col_gate_stage)
        )
    for sX_tile, sO_row_tile, mS_row_stage, sO_col_tile, mS_col_stage in halves:
        quantize_mxfp8_smem_tile_general(
            sX_tile,
            None,
            sO_row_tile,
            mS_row_stage,
            sO_col_tile,
            mS_col_stage,
            None,
            tile_row_start,
            tile_col_start,
            M,
            N,
            ROWWISE=ROWWISE,
            COLWISE=COLWISE,
            WITH_AMAX=False,
            WITH_ACT=False,
            WITH_DACT=False,
            ACTIVATION=None,
            MAX_NORM_RCP=MAX_NORM_RCP,
            DTYPE=DTYPE,
            FP8_DTYPE=FP8_DTYPE,
            WITH_GEMM_SWIZZLED_SCALES=WITH_GEMM_SWIZZLED_SCALES,
            TILE_X=TILE_X,
            TILE_Y=TILE_Y,
            # The gated op maps the zeros TMA fills out-of-bounds elements with to zero.
            SKIP_INPUT_MASKING=True,
            SKIP_SCALE_BOUNDS=False,
            CACHE_ACTIVATION=False,
            DBIAS_REDUCTION_IN_COLWISE=False,
            COLWISE_DBIAS_REDUCTION_ONLY=False,
            DBIAS_REDUCTION_IN_ROWWISE=False,
        )


@cute.jit
def _column_slice(mT: cute.Tensor, col_offset: Int32, num_cols: Int32) -> cute.Tensor:
    """(rows, num_cols) view of the row-major mT starting at column col_offset.

    The pointer is rebuilt rather than offset so that TMA sees it 16-byte aligned, which the C++
    dispatcher guarantees for both the tensor and the column offset of every slice.
    """
    ptr = cute.make_ptr(
        mT.element_type,
        mT.iterator.toint() + Int64(col_offset) * Int64(mT.element_type.width // 8),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return cute.make_tensor(
        ptr, cute.make_layout((mT.shape[0], num_cols), stride=(mT.stride[0], 1))
    )


def _fake_gmem_tensor(dtype, shape, align):
    """Row-major fake gmem tensor for cute.compile."""
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        memspace=cute.AddressSpace.gmem,
        assumed_align=align,
    )


class MXFP8GatedQuantizeConfig(MXFP8QuantizeConfig):
    """Compile-time config of gated_mxfp8.cuh's port: a plain MXFP8 cast config plus the gated op."""

    def __init__(
        self,
        dtype: str,
        fp8_dtype: str,
        rowwise: bool,
        colwise: bool,
        with_gemm_swizzled_scales: bool,
        is_bwd: bool,
        activation: str,
    ):
        super().__init__(
            dtype=dtype,
            fp8_dtype=fp8_dtype,
            rowwise=rowwise,
            colwise=colwise,
            with_gemm_swizzled_scales=with_gemm_swizzled_scales,
            with_amax=False,
            activation="none",
        )
        if activation not in GATED_ACTIVATIONS:
            raise ValueError(
                f"unknown gated activation {activation!r}; expected one of"
                f" {sorted(GATED_ACTIVATIONS)}"
            )
        self.IS_BWD = is_bwd
        self.GATED_ACTIVATION = activation

    def __str__(self):
        return (
            f"MXFP8GatedQuantizeConfig(dtype={self.DTYPE_STR}, fp8_dtype={self.FP8_DTYPE_STR}, "
            f"rowwise={self.ROWWISE}, colwise={self.COLWISE}, "
            f"swizzled={self.WITH_GEMM_SWIZZLED_SCALES}, is_bwd={self.IS_BWD}, "
            f"activation={self.GATED_ACTIVATION})"
        )

    __repr__ = __str__

    def compile(self):
        """Return the compiled CuTeDSL function object for this config."""
        sym_M = cute.sym_int32()
        # The C++ dispatcher requires N % 32 == 0, like the CUDA kernel.
        sym_N = cute.sym_int32(divisibility=32)
        sym_2N = cute.sym_int32(divisibility=64)
        out_cols = sym_2N if self.IS_BWD else sym_N
        # Padded scale shapes, see MXFP8QuantizeConfig.compile
        scale_rowwise_shape = (cute.sym_int32(divisibility=128), cute.sym_int32(divisibility=4))
        scale_colwise_shape = (cute.sym_int32(divisibility=4), cute.sym_int32(divisibility=128))
        params = [Float32(0.0)] * 5  # limit, alpha, glu_linear_offset, beta1, beta2
        return cute.compile(
            MXFP8GatedQuantizeG2SKernel(self),
            _fake_gmem_tensor(self.DTYPE, (sym_M, sym_2N), 16),  # mGatedInput
            _fake_gmem_tensor(self.DTYPE, (sym_M, sym_N), 16) if self.IS_BWD else None,  # mGrad
            # mO_row, mS_row
            _fake_gmem_tensor(self.FP8_DTYPE, (sym_M, out_cols), 16) if self.ROWWISE else None,
            (
                _fake_gmem_tensor(cutlass.Float8E8M0FNU, scale_rowwise_shape, 4)
                if self.ROWWISE
                else None
            ),
            # mO_col, mS_col
            _fake_gmem_tensor(self.FP8_DTYPE, (sym_M, out_cols), 16) if self.COLWISE else None,
            (
                _fake_gmem_tensor(cutlass.Float8E8M0FNU, scale_colwise_shape, 4)
                if self.COLWISE
                else None
            ),
            *params,
            cute.runtime.make_fake_stream(),
            options="--enable-tvm-ffi",
        )


class MXFP8GatedQuantizeG2SKernel(MXFP8QuantizeGeneralG2SKernel):
    """gated_mxfp8.cuh in CuTeDSL. The general kernel's pipeline, loading the act and gate halves
    (and grad, backward) each stage and storing one (forward) or two (backward) quantized halves.

    Inherits the plain-cast tile config, which is gated_mxfp8.cuh's: 64 threads per CTA, each CTA
    covering two vertically stacked 32x64 tiles.
    """

    def __init__(self, cfg: MXFP8GatedQuantizeConfig):
        super().__init__(cfg)
        # Same occupancy hint as the general kernel with a fused activation
        self._MIN_BLOCKS_PER_MP = 1

    @cute.jit
    def __call__(
        self,
        mGatedInput: cute.Tensor,  # [M, 2N] = [act | gate]
        mGrad: Optional[cute.Tensor],  # [M, N], bwd only
        mO_row: Optional[cute.Tensor],  # [M, N] (fwd) or [M, 2N] (bwd)
        mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        limit: Float32,  # ClampedSwiGLUParam
        alpha: Float32,
        glu_linear_offset: Float32,
        beta1: Float32,  # SiTUGLUParam
        beta2: Float32,
        stream: CUstream,
    ):
        cfg = self.cfg
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(f"[CuTeDSL] MXFP8GatedQuantizeG2SKernel.__call__() with config: {cfg}\n")

        mO = mO_row if cfg.ROWWISE else mO_col
        M = mGatedInput.shape[0]
        N = mGrad.shape[1] if cfg.IS_BWD else mO.shape[1]

        # Scales cover the whole output: [M, 2N] backward, with the gate half's scales starting
        # at scale column N / 32 (rowwise) or N (colwise).
        if cutlass.const_expr(cfg.WITH_GEMM_SWIZZLED_SCALES):
            mS_row, mS_col = derive_swizzled_scale_layout(
                M, mO.shape[1], cfg.ROWWISE, cfg.COLWISE, mS_row, mS_col
            )
        mS_row_gate = None
        mS_col_gate = None
        mO_row_gate = None
        mO_col_gate = None
        if cutlass.const_expr(cfg.IS_BWD):
            # Swizzled scales are only translation invariant across whole 128-column tiles,
            # which the C++ dispatcher guarantees by requiring N % 128 == 0 for them.
            if cutlass.const_expr(cfg.ROWWISE):
                mS_row_gate = cute.domain_offset((0, N // MXFP8_BLOCK_SCALING_SIZE), mS_row)
                mO_row_gate = _column_slice(mO_row, N, N)
                mO_row = _column_slice(mO_row, 0, N)
            if cutlass.const_expr(cfg.COLWISE):
                mS_col_gate = cute.domain_offset((0, N), mS_col)
                mO_col_gate = _column_slice(mO_col, N, N)
                mO_col = _column_slice(mO_col, 0, N)

        atoms = self._make_tma_atoms(
            mGrad,
            _column_slice(mGatedInput, 0, N),
            _column_slice(mGatedInput, N, N),
            mO_row,
            mO_row_gate,
            mO_col,
            mO_col_gate,
        )
        # The TMA atoms are spread from a list, which pylint cannot count
        self.kernel(  # pylint: disable=no-value-for-parameter
            M,
            N,
            mS_row,
            mS_row_gate,
            mS_col,
            mS_col_gate,
            limit,
            alpha,
            glu_linear_offset,
            beta1,
            beta2,
            *atoms,
        ).launch(
            grid=self._grid(M, N),
            block=[self._THREADS_PER_CTA],
            min_blocks_per_mp=self._MIN_BLOCKS_PER_MP,
            stream=stream,
        )

    def _grid(self, M: Int32, N: Int32):
        return [
            cute.ceil_div(N, self._TILE_COLS),
            cute.ceil_div(M, self._TILE_ROWS * self._NUM_TILES),
        ]

    @cute.jit
    def _make_tma_atoms(
        self,
        mGrad: Optional[cute.Tensor],
        mAct: cute.Tensor,
        mGate: cute.Tensor,
        mO_row_act: Optional[cute.Tensor],
        mO_row_gate: Optional[cute.Tensor],
        mO_col_act: Optional[cute.Tensor],
        mO_col_gate: Optional[cute.Tensor],
    ):
        """TMA atoms and their gmem views for every (TILE_ROWS, TILE_COLS) operand; None if absent."""
        smem_tile_layout = cute.make_ordered_layout(
            (self._TILE_ROWS, self._TILE_COLS), order=(1, 0)
        )
        cta_tiler = (self._TILE_ROWS, self._TILE_COLS)
        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        atoms = []
        for op, mT in (
            (op_load, mGrad),
            (op_load, mAct),
            (op_load, mGate),
            (op_store, mO_row_act),
            (op_store, mO_row_gate),
            (op_store, mO_col_act),
            (op_store, mO_col_gate),
        ):
            if cutlass.const_expr(mT is None):
                atoms.extend([None, None])
            else:
                atoms.extend(
                    cute.nvgpu.cpasync.make_tiled_tma_atom(
                        op, mT, smem_tile_layout, cta_tiler, num_multicast=1
                    )
                )
        return atoms

    @cute.jit
    def _copy_inputs_to_stage(
        self,
        tile_y: Int32,
        tile_x: Int32,
        stage: Int32,
        tma_bar_ptr: cute.Pointer,
        tma_atom_grad: Optional[cute.CopyAtom],
        tXgGrad: Optional[cute.Tensor],
        tXsGrad: Optional[cute.Tensor],
        tma_atom_act: cute.CopyAtom,
        tXgAct: cute.Tensor,
        tXsAct: cute.Tensor,
        tma_atom_gate: cute.CopyAtom,
        tXgGate: cute.Tensor,
        tXsGate: cute.Tensor,
    ):
        """Issue this tile's input TMA loads into the pipeline stage, all on one barrier."""
        if cutlass.const_expr(self.cfg.IS_BWD):
            cute.copy(
                tma_atom_grad,
                tXgGrad[(None, (tile_y, tile_x))],
                tXsGrad[(None, stage)],
                tma_bar_ptr=tma_bar_ptr,
            )
        cute.copy(
            tma_atom_act,
            tXgAct[(None, (tile_y, tile_x))],
            tXsAct[(None, stage)],
            tma_bar_ptr=tma_bar_ptr,
        )
        cute.copy(
            tma_atom_gate,
            tXgGate[(None, (tile_y, tile_x))],
            tXsGate[(None, stage)],
            tma_bar_ptr=tma_bar_ptr,
        )

    @cute.kernel
    def kernel(
        self,
        M: Int32,
        N: Int32,
        mS_row_act: Optional[cute.Tensor],
        mS_row_gate: Optional[cute.Tensor],
        mS_col_act: Optional[cute.Tensor],
        mS_col_gate: Optional[cute.Tensor],
        limit: Float32,
        alpha: Float32,
        glu_linear_offset: Float32,
        beta1: Float32,
        beta2: Float32,
        tma_atom_grad: Optional[cute.CopyAtom],
        tma_src_grad: Optional[cute.Tensor],
        tma_atom_act: cute.CopyAtom,
        tma_src_act: cute.Tensor,
        tma_atom_gate: cute.CopyAtom,
        tma_src_gate: cute.Tensor,
        tma_atom_row_act: Optional[cute.CopyAtom],
        tma_dst_row_act: Optional[cute.Tensor],
        tma_atom_row_gate: Optional[cute.CopyAtom],
        tma_dst_row_gate: Optional[cute.Tensor],
        tma_atom_col_act: Optional[cute.CopyAtom],
        tma_dst_col_act: Optional[cute.Tensor],
        tma_atom_col_gate: Optional[cute.CopyAtom],
        tma_dst_col_gate: Optional[cute.Tensor],
    ):
        cfg = self.cfg
        DTYPE = cfg.DTYPE
        TILE_ROWS = self._TILE_ROWS
        TILE_COLS = self._TILE_COLS
        NUM_STAGES = self._NUM_STAGES
        NUM_TILES = self._NUM_TILES

        # Scale views tiled like the general kernel's
        row_scale_tile = (TILE_ROWS, TILE_COLS // MXFP8_BLOCK_SCALING_SIZE)
        col_scale_tile = (TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE, TILE_COLS)
        if cutlass.const_expr(cfg.ROWWISE):
            mS_row_act = cute.zipped_divide(mS_row_act, row_scale_tile)
            if cutlass.const_expr(cfg.IS_BWD):
                mS_row_gate = cute.zipped_divide(mS_row_gate, row_scale_tile)
        if cutlass.const_expr(cfg.COLWISE):
            mS_col_act = cute.zipped_divide(mS_col_act, col_scale_tile)
            if cutlass.const_expr(cfg.IS_BWD):
                mS_col_gate = cute.zipped_divide(mS_col_gate, col_scale_tile)

        # Shared memory: one ((TILE_ROWS, TILE_COLS), NUM_STAGES) ring buffer per operand
        smem = cutlass.memory.SmemAllocator()

        @cute.struct
        class BarrierStorage:  # pylint: disable=missing-class-docstring
            mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]

        storage = smem.allocate(BarrierStorage)
        stage_layout = cute.make_layout(
            ((TILE_ROWS, TILE_COLS), NUM_STAGES),
            stride=((TILE_COLS, 1), TILE_ROWS * TILE_COLS),
        )
        sAct = smem.allocate_tensor(DTYPE, stage_layout, byte_alignment=128)
        sGate = smem.allocate_tensor(DTYPE, stage_layout, byte_alignment=128)
        sGrad = None
        if cutlass.const_expr(cfg.IS_BWD):
            sGrad = smem.allocate_tensor(DTYPE, stage_layout, byte_alignment=128)
        sO_row_act = None
        sO_row_gate = None
        sO_col_act = None
        sO_col_gate = None
        if cutlass.const_expr(cfg.ROWWISE):
            sO_row_act = smem.allocate_tensor(cfg.FP8_DTYPE, stage_layout, byte_alignment=128)
            if cutlass.const_expr(cfg.IS_BWD):
                sO_row_gate = smem.allocate_tensor(cfg.FP8_DTYPE, stage_layout, byte_alignment=128)
        if cutlass.const_expr(cfg.COLWISE):
            sO_col_act = smem.allocate_tensor(cfg.FP8_DTYPE, stage_layout, byte_alignment=128)
            if cutlass.const_expr(cfg.IS_BWD):
                sO_col_gate = smem.allocate_tensor(cfg.FP8_DTYPE, stage_layout, byte_alignment=128)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()

        if warp_idx == 0:
            if cutlass.const_expr(cfg.IS_BWD):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_grad)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_act)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_gate)

        # Warp 0's elected thread produces (issues TMA), every warp consumes
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self._NUM_WARPS)
        num_inputs = 3 if cfg.IS_BWD else 2
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_storage.data_ptr(),
            num_stages=NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            # Every input tile of a stage arrives on the same barrier
            tx_count=num_inputs * TILE_ROWS * TILE_COLS * DTYPE.width // 8,
            cta_layout_vmnk=None,
        )
        prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NUM_STAGES)
        cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, NUM_STAGES)

        # Partition every operand for TMA (single CTA, no multicast)
        cta_tiler = (TILE_ROWS, TILE_COLS)
        tXsGrad = None
        tXgGrad = None
        if cutlass.const_expr(cfg.IS_BWD):
            tXsGrad, tXgGrad = cute.nvgpu.cpasync.tma_partition(
                tma_atom_grad,
                0,
                cute.make_layout(1),
                sGrad,
                cute.zipped_divide(tma_src_grad, cta_tiler),
            )
        tXsAct, tXgAct = cute.nvgpu.cpasync.tma_partition(
            tma_atom_act, 0, cute.make_layout(1), sAct, cute.zipped_divide(tma_src_act, cta_tiler)
        )
        tXsGate, tXgGate = cute.nvgpu.cpasync.tma_partition(
            tma_atom_gate,
            0,
            cute.make_layout(1),
            sGate,
            cute.zipped_divide(tma_src_gate, cta_tiler),
        )
        # Output (atom, smem partition, gmem partition) triples, act half first
        outputs = []
        for atom_out, dst_out, sO_out in (
            (tma_atom_row_act, tma_dst_row_act, sO_row_act),
            (tma_atom_row_gate, tma_dst_row_gate, sO_row_gate),
            (tma_atom_col_act, tma_dst_col_act, sO_col_act),
            (tma_atom_col_gate, tma_dst_col_gate, sO_col_gate),
        ):
            if cutlass.const_expr(atom_out is not None):
                tXsO_out, tXgO_out = cute.nvgpu.cpasync.tma_partition(
                    atom_out,
                    0,
                    cute.make_layout(1),
                    sO_out,
                    cute.zipped_divide(dst_out, cta_tiler),
                )
                outputs.append((atom_out, tXsO_out, tXgO_out))

        # Barrier init must be visible to every thread
        cute.arch.sync_threads()

        # Prologue: fill the pipeline
        if warp_idx == 0:
            for s in cutlass.range_constexpr(min(NUM_STAGES, NUM_TILES)):
                mainloop_pipeline.producer_acquire(prod_state)
                self._copy_inputs_to_stage(
                    bidy * NUM_TILES + s,
                    bidx,
                    prod_state.index,
                    mainloop_pipeline.producer_get_barrier(prod_state),
                    tma_atom_grad,
                    tXgGrad,
                    tXsGrad,
                    tma_atom_act,
                    tXgAct,
                    tXsAct,
                    tma_atom_gate,
                    tXgGate,
                    tXsGate,
                )
                mainloop_pipeline.producer_commit(prod_state)
                prod_state.advance()

        for tile_idx in cutlass.range(NUM_TILES, unroll=1):
            mainloop_pipeline.consumer_wait(cons_state)
            # This tile reuses the output buffers of the tile NUM_STAGES ago; wait until TMA has
            # read them.
            if cutlass.const_expr(NUM_TILES > NUM_STAGES):
                if warp_idx == 0:
                    cute.arch.cp_async_bulk_wait_group(NUM_STAGES - 1, read=True)
                cute.arch.sync_threads()
            stage_idx = cons_state.index
            tile_idx_y = bidy * NUM_TILES + tile_idx
            scale_coord = (None, (tile_idx_y, bidx))

            sGrad_tile = None
            if cutlass.const_expr(cfg.IS_BWD):
                sGrad_tile = sGrad[(None, stage_idx)]
            sO_row_act_tile = None
            mS_row_act_stage = None
            sO_row_gate_tile = None
            mS_row_gate_stage = None
            if cutlass.const_expr(cfg.ROWWISE):
                sO_row_act_tile = sO_row_act[(None, stage_idx)]
                mS_row_act_stage = cute.flatten(mS_row_act[scale_coord])
                if cutlass.const_expr(cfg.IS_BWD):
                    sO_row_gate_tile = sO_row_gate[(None, stage_idx)]
                    mS_row_gate_stage = cute.flatten(mS_row_gate[scale_coord])
            sO_col_act_tile = None
            mS_col_act_stage = None
            sO_col_gate_tile = None
            mS_col_gate_stage = None
            if cutlass.const_expr(cfg.COLWISE):
                sO_col_act_tile = sO_col_act[(None, stage_idx)]
                mS_col_act_stage = cute.flatten(mS_col_act[scale_coord])
                if cutlass.const_expr(cfg.IS_BWD):
                    sO_col_gate_tile = sO_col_gate[(None, stage_idx)]
                    mS_col_gate_stage = cute.flatten(mS_col_gate[scale_coord])

            quantize_gated_mxfp8_smem_tile(
                sAct[(None, stage_idx)],
                sGate[(None, stage_idx)],
                sGrad_tile,
                sO_row_act_tile,
                mS_row_act_stage,
                sO_row_gate_tile,
                mS_row_gate_stage,
                sO_col_act_tile,
                mS_col_act_stage,
                sO_col_gate_tile,
                mS_col_gate_stage,
                tile_idx_y * TILE_ROWS,
                bidx * TILE_COLS,
                M,
                N,
                limit,
                alpha,
                glu_linear_offset,
                beta1,
                beta2,
                ACTIVATION=cfg.GATED_ACTIVATION,
                IS_BWD=cfg.IS_BWD,
                ROWWISE=cfg.ROWWISE,
                COLWISE=cfg.COLWISE,
                MAX_NORM_RCP=cfg.MAX_NORM_RCP,
                DTYPE=DTYPE,
                FP8_DTYPE=cfg.FP8_DTYPE,
                WITH_GEMM_SWIZZLED_SCALES=cfg.WITH_GEMM_SWIZZLED_SCALES,
                TILE_X=TILE_COLS,
                TILE_Y=TILE_ROWS,
            )

            # Make the SMEM writes visible to the TMA's async proxy before it reads them
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            mainloop_pipeline.consumer_release(cons_state)

            if warp_idx == 0:
                # Output tensors are only read here, inside this dynamic branch, so these loop
                # targets are fresh names rather than rebindings of values defined outside it.
                for store_atom, store_sO, store_gO in outputs:
                    cute.copy(
                        store_atom,
                        store_sO[(None, stage_idx)],
                        store_gO[(None, (tile_idx_y, bidx))],
                    )
                cute.arch.cp_async_bulk_commit_group()

            cons_state.advance()

            # Refill the stage just released with the tile NUM_STAGES ahead
            if warp_idx == 0:
                next_tile_idx = tile_idx + NUM_STAGES
                if next_tile_idx < NUM_TILES:
                    mainloop_pipeline.producer_acquire(prod_state)
                    self._copy_inputs_to_stage(
                        bidy * NUM_TILES + next_tile_idx,
                        bidx,
                        prod_state.index,
                        mainloop_pipeline.producer_get_barrier(prod_state),
                        tma_atom_grad,
                        tXgGrad,
                        tXsGrad,
                        tma_atom_act,
                        tXgAct,
                        tXsAct,
                        tma_atom_gate,
                        tXgGate,
                        tXsGate,
                    )
                    mainloop_pipeline.producer_commit(prod_state)
                    prod_state.advance()

        # Only wait until the in-flight TMA stores have read SMEM, so the CTA retires early
        cute.arch.cp_async_bulk_wait_group(0, read=True)
