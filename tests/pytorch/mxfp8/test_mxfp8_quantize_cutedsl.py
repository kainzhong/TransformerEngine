# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for MXFP8 quantization via CuTeDSL kernel.

Validates that the CuTeDSL implementation produces bit-identical results
to the reference C++ MXFP8 quantizer in Transformer Engine. Covers:

  - plain quantize (rowwise / colwise / both)         vs. `tex.quantize`
  - quantize -> dequantize roundtrip accuracy
  - IS_ACT  (fused forward activation)                vs. `tex.{relu,gelu,silu}`
  - IS_DACT (fused backward activation)               vs. `tex.{drelu,dgelu,dsilu}`
  - IS_DACT + IS_DBIAS (act gradient + per-col sum)   vs. `tex.dbias_d{relu,gelu,silu}`
"""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage

from quantize_mxfp8_cutedsl_alt import (
    quantize_mxfp8_cutedsl,
    MXFP8_BLOCK_SIZE,
)

from mxfp8_utils import get_mxfp8_scale_shape_no_padding

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def unpack_quantized_tensor(quantized_tensor: MXFP8TensorStorage):
    """Extract components from a quantized MXFP8 tensor."""
    qx, sx, qx_t, sx_t = None, None, None, None
    if quantized_tensor._rowwise_data is not None:
        qx = quantized_tensor._rowwise_data.view(dtype=torch.uint8)
    if quantized_tensor._rowwise_scale_inv is not None:
        sx = quantized_tensor._rowwise_scale_inv
    if quantized_tensor._columnwise_data is not None:
        qx_t = quantized_tensor._columnwise_data.view(dtype=torch.uint8)
    if quantized_tensor._columnwise_scale_inv is not None:
        sx_t = quantized_tensor._columnwise_scale_inv
    return qx, sx, qx_t, sx_t


def _dirs(name):
    return {"row": (True, False), "col": (False, True), "both": (True, True)}[name]


def _assert_quant_equal(dsl, ref_tensor, rowwise, colwise, tag=""):
    """Bit-exact compare DSL output dict against a TE quantized tensor."""
    if rowwise:
        ref_d = ref_tensor._rowwise_data.view(torch.uint8)
        ref_s = ref_tensor._rowwise_scale_inv.view(torch.uint8)
        assert torch.equal(dsl["rowwise_data"], ref_d), f"{tag}: rowwise data mismatch"
        assert torch.equal(dsl["rowwise_scale"], ref_s), f"{tag}: rowwise scale mismatch"
    if colwise:
        ref_d = ref_tensor._columnwise_data.view(torch.uint8)
        ref_s = ref_tensor._columnwise_scale_inv.view(torch.uint8)
        assert torch.equal(dsl["colwise_data"], ref_d), f"{tag}: colwise data mismatch"
        assert torch.equal(dsl["colwise_scale"], ref_s), f"{tag}: colwise scale mismatch"


# ---------------------------------------------------------------------------
# Plain quantize (no activation, no dbias)
# ---------------------------------------------------------------------------


def check_mxfp8_cutedsl_vs_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_rowwise: bool,
    return_transpose: bool,
) -> None:
    """Compare CuTeDSL quantization against the reference TE quantizer."""
    te_dtype = tex.DType.kFloat8E4M3

    device = "cuda"
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x = torch.randn((M, N), dtype=x_dtype, device=device)

    # --- Reference: TE C++ quantizer ---
    quantizer = MXFP8Quantizer(
        fp8_dtype=te_dtype,
        rowwise=return_rowwise,
        columnwise=return_transpose,
    )
    ref_result = quantizer(x)
    ref_qx, ref_sx, ref_qx_t, ref_sx_t = unpack_quantized_tensor(ref_result)

    # --- CuTeDSL quantizer ---
    dsl_result = quantize_mxfp8_cutedsl(
        x,
        fp8_dtype="e4m3",
        rowwise=return_rowwise,
        colwise=return_transpose,
    )
    torch.cuda.synchronize()

    # --- Compare rowwise ---
    if return_rowwise:
        dsl_qx = dsl_result["rowwise_data"]
        dsl_sx = dsl_result["rowwise_scale"]

        expected_scale_shape = get_mxfp8_scale_shape_no_padding(x.shape, columnwise=False)
        assert dsl_qx.shape == ref_qx.shape, (
            f"Rowwise data shape mismatch: DSL={dsl_qx.shape} vs Ref={ref_qx.shape}"
        )
        assert dsl_sx.shape == expected_scale_shape, (
            f"Rowwise scale shape mismatch: DSL={dsl_sx.shape} vs expected={expected_scale_shape}"
        )

        torch.testing.assert_close(
            dsl_qx, ref_qx, atol=0.0, rtol=0.0,
            msg="Rowwise quantized data mismatch between CuTeDSL and reference",
        )

        ref_sx_u8 = ref_sx.view(dtype=torch.uint8)
        torch.testing.assert_close(
            dsl_sx, ref_sx_u8, atol=0.0, rtol=0.0,
            msg="Rowwise scale mismatch between CuTeDSL and reference",
        )

    # --- Compare colwise ---
    if return_transpose:
        dsl_qx_t = dsl_result["colwise_data"]
        dsl_sx_t = dsl_result["colwise_scale"]

        expected_scale_shape_t = get_mxfp8_scale_shape_no_padding(x.shape, columnwise=True)
        assert dsl_qx_t.shape == ref_qx_t.shape, (
            f"Colwise data shape mismatch: DSL={dsl_qx_t.shape} vs Ref={ref_qx_t.shape}"
        )
        assert dsl_sx_t.shape == expected_scale_shape_t, (
            f"Colwise scale shape mismatch: DSL={dsl_sx_t.shape} vs expected={expected_scale_shape_t}"
        )

        torch.testing.assert_close(
            dsl_qx_t, ref_qx_t, atol=0.0, rtol=0.0,
            msg="Colwise quantized data mismatch between CuTeDSL and reference",
        )

        ref_sx_t_u8 = ref_sx_t.view(dtype=torch.uint8)
        torch.testing.assert_close(
            dsl_sx_t, ref_sx_t_u8, atol=0.0, rtol=0.0,
            msg="Colwise scale mismatch between CuTeDSL and reference",
        )


def check_mxfp8_cutedsl_roundtrip(
    x_dtype: torch.dtype,
    M: int,
    N: int,
) -> None:
    """Verify that quantize -> dequantize produces reasonable results."""
    device = "cuda"
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x = torch.randn((M, N), dtype=x_dtype, device=device)

    result = quantize_mxfp8_cutedsl(x, fp8_dtype="e4m3", rowwise=True)
    torch.cuda.synchronize()

    qx = result["rowwise_data"]
    sx = result["rowwise_scale"]

    x_f32 = x.float()
    qx_f32 = qx.view(torch.float8_e4m3fn).float()

    sx_float = torch.pow(2.0, sx.float().unsqueeze(-1) - 127.0)
    sx_float = sx_float.expand(M, N // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE).reshape(M, N)

    dequantized = qx_f32 * sx_float

    rel_err = (dequantized - x_f32).abs() / (x_f32.abs() + 1e-10)
    mean_rel_err = rel_err.mean().item()

    assert mean_rel_err < 0.15, (
        f"Round-trip relative error too high: {mean_rel_err:.4f}"
    )


# ---------------------------------------------------------------------------
# Pytest test cases — plain quantize
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 128),
        (256, 256),
        (1024, 256),
        (256, 1024),
        (1024, 1024),
        (8192, 1024),
        (16384, 8192),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "quantize_mode",
    ["rowwise_only", "colwise_only", "both_directions"],
)
def test_mxfp8_cutedsl_vs_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    quantize_mode: str,
) -> None:
    """Test CuTeDSL MXFP8 quantization matches the reference C++ implementation."""
    if quantize_mode == "rowwise_only":
        return_rowwise, return_transpose = True, False
    elif quantize_mode == "colwise_only":
        return_rowwise, return_transpose = False, True
    elif quantize_mode == "both_directions":
        return_rowwise, return_transpose = True, True
    else:
        raise ValueError(f"Invalid quantize mode: {quantize_mode}")

    check_mxfp8_cutedsl_vs_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        return_rowwise=return_rowwise,
        return_transpose=return_transpose,
    )


@pytest.mark.parametrize(
    "M, N",
    [
        (128, 128),
        (1024, 1024),
        (4096, 2048),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float16], ids=str)
def test_mxfp8_cutedsl_roundtrip(
    x_dtype: torch.dtype,
    M: int,
    N: int,
) -> None:
    """Test that CuTeDSL MXFP8 quantize -> dequantize roundtrip is accurate."""
    check_mxfp8_cutedsl_roundtrip(x_dtype=x_dtype, M=M, N=N)


# ---------------------------------------------------------------------------
# Pytest test cases — GEMM-swizzled scales (WITH_GEMM_SWIZZLED_SCALES)
# ---------------------------------------------------------------------------
#
# When `with_gemm_swizzled_scales=True`, the kernel writes the per-block E8M0
# scale bytes to the cuBLAS MXFP8 swizzled layout
# (https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout) so that
# downstream MXFP8 GEMM can consume them directly without a separate swizzle
# pass. The FP8 *data* bytes are unchanged — only the scale memory permutation
# differs. The reference is TE's `MXFP8Quantizer` with `optimize_for_gemm=True`.
#
# Constraints: M and N must be multiples of 128 (the swizzle tile is 128 in
# the row axis and 128 = 4 scale blocks in the col axis).


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 1024),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("direction", ["row", "col", "both"])
def test_mxfp8_cutedsl_swizzled_scales(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    direction: str,
) -> None:
    """WITH_GEMM_SWIZZLED_SCALES: scale bytes laid out in cuBLAS GEMM-swizzled
    order match TE's `MXFP8Quantizer(..., optimize_for_gemm=True)` byte-for-byte."""
    rowwise, colwise = _dirs(direction)
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=x_dtype, device="cuda")

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise, columnwise=colwise,
    )
    quantizer.optimize_for_gemm = True
    ref = tex.quantize(x, quantizer)

    dsl = quantize_mxfp8_cutedsl(
        x, rowwise=rowwise, colwise=colwise,
        with_gemm_swizzled_scales=True,
    )
    torch.cuda.synchronize()

    _assert_quant_equal(dsl, ref, rowwise, colwise,
                        tag=f"swizzle/{direction}/{x_dtype}/{M}x{N}")


# ---------------------------------------------------------------------------
# Pytest test cases — fused activation combos
# ---------------------------------------------------------------------------


_COMBO_SHAPES = [(128, 128), (256, 256), (512, 256)]
_COMBO_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
_DIRECTIONS = ["row", "col", "both"]


_TE_FWD_ACT = {"relu": tex.relu, "gelu": tex.gelu, "silu": tex.silu}
_TE_BWD_ACT = {"drelu": tex.drelu, "dgelu": tex.dgelu, "dsilu": tex.dsilu}
_TE_DBIAS_DACT = {
    "drelu": tex.dbias_drelu,
    "dgelu": tex.dbias_dgelu,
    "dsilu": tex.dbias_dsilu,
}


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", _COMBO_SHAPES)
@pytest.mark.parametrize("x_dtype", _COMBO_DTYPES, ids=str)
@pytest.mark.parametrize("activation", sorted(_TE_FWD_ACT))
@pytest.mark.parametrize("direction", _DIRECTIONS)
def test_mxfp8_cutedsl_activation(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    activation: str,
    direction: str,
) -> None:
    """IS_ACT path: fused forward activation + MXFP8 quantize vs. tex.{relu,gelu,silu}."""
    rowwise, colwise = _dirs(direction)
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=x_dtype, device="cuda")

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise, columnwise=colwise,
    )
    ref = _TE_FWD_ACT[activation](x, quantizer)

    dsl = quantize_mxfp8_cutedsl(
        x, rowwise=rowwise, colwise=colwise, activation=activation,
    )
    torch.cuda.synchronize()

    _assert_quant_equal(dsl, ref, rowwise, colwise,
                        tag=f"{activation}/{direction}/{x_dtype}")


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", _COMBO_SHAPES)
@pytest.mark.parametrize("x_dtype", _COMBO_DTYPES, ids=str)
@pytest.mark.parametrize("activation", sorted(_TE_BWD_ACT))
@pytest.mark.parametrize("direction", _DIRECTIONS)
def test_mxfp8_cutedsl_dact(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    activation: str,
    direction: str,
) -> None:
    """IS_DACT path: fused backward activation + MXFP8 quantize vs. tex.{drelu,dgelu,dsilu}."""
    rowwise, colwise = _dirs(direction)
    torch.manual_seed(0)
    grad = torch.randn((M, N), dtype=x_dtype, device="cuda")
    act_in = torch.randn((M, N), dtype=x_dtype, device="cuda")

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise, columnwise=colwise,
    )
    ref = _TE_BWD_ACT[activation](grad, act_in, quantizer)

    dsl = quantize_mxfp8_cutedsl(
        grad, rowwise=rowwise, colwise=colwise,
        activation=activation, act_input=act_in,
        is_dact=True
    )
    torch.cuda.synchronize()

    _assert_quant_equal(dsl, ref, rowwise, colwise,
                        tag=f"{activation}/{direction}/{x_dtype}")


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", _COMBO_SHAPES)
@pytest.mark.parametrize("x_dtype", _COMBO_DTYPES, ids=str)
@pytest.mark.parametrize("activation", sorted(_TE_DBIAS_DACT))
# Cover both IS_DBIAS paths: path A (colwise piggyback, used for "col" & "both")
# and path B (rowwise-only shmem-transpose, used for "row").
@pytest.mark.parametrize("direction", _DIRECTIONS)
def test_mxfp8_cutedsl_dbias_dact(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    activation: str,
    direction: str,
) -> None:
    """IS_DACT + IS_DBIAS: fused dact + per-column dbias vs. tex.dbias_d{relu,gelu,silu}."""
    rowwise, colwise = _dirs(direction)
    torch.manual_seed(0)
    grad = torch.randn((M, N), dtype=x_dtype, device="cuda")
    act_in = torch.randn((M, N), dtype=x_dtype, device="cuda")

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise, columnwise=colwise,
    )
    # TE returns (dbias, quant_tensor)
    ref_dbias, ref_quant = _TE_DBIAS_DACT[activation](grad, act_in, quantizer)

    dsl = quantize_mxfp8_cutedsl(
        grad, rowwise=rowwise, colwise=colwise,
        activation=activation, act_input=act_in,
        compute_dbias=True, is_dact=True,
    )
    torch.cuda.synchronize()

    tag = f"{activation}/{direction}/{x_dtype}"
    _assert_quant_equal(dsl, ref_quant, rowwise, colwise, tag=tag)
    assert torch.equal(dsl["dbias"], ref_dbias), (
        f"{tag}: dbias mismatch — "
        f"{(dsl['dbias'] != ref_dbias).sum().item()}/{N} diffs, "
        f"max |Δ|={(dsl['dbias'].float() - ref_dbias.float()).abs().max().item():.6g}"
    )
