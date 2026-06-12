# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

from transformer_engine.common.cutedsl.cutedsl_utils import str_to_te_dtype
import transformer_engine.pytorch  # noqa: F401  (loads libtransformer_engine.so)
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.common.cutedsl.cast.mxfp8_quantization import (
    get_mxfp8_quantizer_jit,
)

MXFP8_BLOCK = 32  # MXFP8 scale block size; valid shapes must be multiples of this.

# 2 aligned (no scale padding) + 2 padded (partial tiles);
SHAPES = [(256, 256), (128, 512), (96, 224), (160, 96)]

def get_dtype_combinations():
    dtype_row = ("e4m3", "e5m2", "none")
    dtype_column = ("e4m3", "e5m2", "none")
    return [(r, c) for r in dtype_row for c in dtype_column]

DTYPE_PAIRS = get_dtype_combinations()

def reference_quantize(x, fp8_type, rowwise, columnwise, swizzle):
    q = MXFP8Quantizer(fp8_dtype=str_to_te_dtype(fp8_type), rowwise=rowwise, columnwise=columnwise)
    q.optimize_for_gemm = swizzle  # makes the native kernel emit swizzled scales
    ref = tex.quantize(x.clone(), q)
    return ref

@pytest.mark.parametrize("swizzle", [False, True])
@pytest.mark.parametrize("dtype_pair", DTYPE_PAIRS)
@pytest.mark.parametrize("shape", SHAPES)
def test_flex_mxfp8_bitexact(shape, dtype_pair, swizzle):
    M, N = shape
    dtype_row, dtype_column = dtype_pair
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    # No direction is invalid -- the quantizer must reject it at construction.
    if dtype_row == "none" and dtype_column == "none":
        with pytest.raises(ValueError):
            get_mxfp8_quantizer_jit(x, dtype_row, dtype_column, with_gemm_swizzled_scales=swizzle)
        return

    flex_q = get_mxfp8_quantizer_jit(
        x, dtype_row=dtype_row, dtype_col=dtype_column, with_gemm_swizzled_scales=swizzle
    )
    # Pure-Python path: flex_q(x) -> quantize_impl -> compiled cute kernel.
    # Returns a HybridQuantizedTensor whose per-direction sub-storages are
    # single-direction MXFP8Tensors.
    flex = flex_q(x)
    torch.cuda.synchronize()

    if dtype_row != "none":
        scale_M, scale_N = M, N // MXFP8_BLOCK
        flex_row = flex._rowwise_storage
        assert flex_row is not None, "row!=none must produce a rowwise sub-storage"
        # Reference for this direction uses THIS direction's dtype.
        ref = reference_quantize(x, dtype_row, rowwise=True, columnwise=False, swizzle=swizzle)
        assert ref._rowwise_data.shape == flex_row._rowwise_data.shape, "rowwise data shape mismatch"
        assert ref._rowwise_scale_inv.shape == flex_row._rowwise_scale_inv.shape, "rowwise scale shape mismatch"
        torch.testing.assert_close(flex_row._rowwise_data, ref._rowwise_data, rtol=0, atol=0)  # bit-identical
        if swizzle:
            torch.testing.assert_close(flex_row._rowwise_scale_inv, ref._rowwise_scale_inv, rtol=0, atol=0)
        else:
            torch.testing.assert_close(
                flex_row._rowwise_scale_inv[:scale_M, :scale_N],
                ref._rowwise_scale_inv[:scale_M, :scale_N],
                rtol=0, atol=0
            )
    else:
        assert flex._rowwise_storage is None, "row=none must not produce rowwise sub-storage"

    if dtype_column != "none":
        scale_M, scale_N = M // MXFP8_BLOCK, N
        flex_col = flex._columnwise_storage
        assert flex_col is not None, "col!=none must produce a columnwise sub-storage"
        ref = reference_quantize(x, dtype_column, rowwise=False, columnwise=True, swizzle=swizzle)
        assert ref._columnwise_data.shape == flex_col._columnwise_data.shape, "columnwise data shape mismatch"
        assert ref._columnwise_scale_inv.shape == flex_col._columnwise_scale_inv.shape, "columnwise scale shape mismatch"
        torch.testing.assert_close(flex_col._columnwise_data, ref._columnwise_data, rtol=0, atol=0)  # bit-identical
        if swizzle:
            torch.testing.assert_close(flex_col._columnwise_scale_inv, ref._columnwise_scale_inv, rtol=0, atol=0)
        else:
            torch.testing.assert_close(
                flex_col._columnwise_scale_inv[:scale_M, :scale_N],
                ref._columnwise_scale_inv[:scale_M, :scale_N],
                rtol=0, atol=0
            )
    else:
        assert flex._columnwise_storage is None, "col=none must not produce colwise sub-storage"
