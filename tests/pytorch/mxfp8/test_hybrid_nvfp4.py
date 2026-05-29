# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Bitwise-exactness tests for the hybrid MXFP8 + NVFP4 CuTeDSL kernel.

`MXFP8QuantizeSmemKernel` can emit MXFP8 in one direction and NVFP4 in the
other, reusing the same shared-memory input tile (single DRAM read). These
tests drive it directly (no tvm-ffi) for both hybrid pairings and check, byte
for byte:

  * the NVFP4 output (data + E4M3 scale) against `NVFP4QuantizerRef`, the
    authoritative pure-PyTorch emulation reference, and
  * the MXFP8 output against the same kernel run MXFP8-only — proving the NVFP4
    addition does not perturb the MXFP8 path.

Run: ``pytest tests/pytorch/mxfp8/test_hybrid_nvfp4.py -v``

Requires a Blackwell GPU (sm_100+): NVFP4's fp4 cvt needs the ``sm_100a``
target, which is set below before cutlass is imported.
"""

import os
import sys


def _detect_arch() -> str:
    # NVFP4 (cvt.e2m1x2.f32) requires the arch-specific "a" target, e.g.
    # sm_100a — plain sm_100 rejects the fp4 cvt at ptxas time.
    try:
        import torch
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        return f"sm_{major}{minor}a"
    except Exception:
        return "sm_100a"


os.environ.setdefault("CUTE_DSL_ARCH", _detect_arch())
sys.path.insert(0, os.path.dirname(__file__))

import pytest
import torch


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


# Everything in this module needs the fp4 cvt -> Blackwell only.
pytestmark = pytest.mark.skipif(
    not _is_blackwell(),
    reason="hybrid NVFP4 kernel requires a Blackwell GPU (sm_100+) for the fp4 cvt",
)

# Imports that pull in cutlass / the kernel are deferred behind the skip guard
# so collection on non-Blackwell / CPU machines doesn't blow up at import time.
if _is_blackwell():
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32
    from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor

    from quantize_mxfp8_cutedsl_alt import (
        MXFP8QuantizeConfig,
        MXFP8QuantizeSmemKernel,
        HybridQuantizeSmemKernel,
        quantize_hybrid_cutedsl,
        SCALE_DIM,            # MXFP8 block (32)
        SCALE_DIM_NVFP4,      # NVFP4 block (16)
    )

    from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import (
        NVFP4QuantizerRef,
    )
    from transformer_engine.pytorch.custom_recipes import utils as cr_utils

FP4_E2M1_MAX = 6.0
FP8E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# Direct (non-tvm-ffi) compile + launch of the hybrid kernel.
# ---------------------------------------------------------------------------
def _fake(dtype, shape, align=16):
    order = tuple(range(len(shape) - 1, -1, -1))  # row-major: 2D -> (1,0), 1D -> (0,)
    return make_fake_compact_tensor(
        dtype, shape, stride_order=order,
        memspace=cute.AddressSpace.gmem, assumed_align=align,
    )


def compute_s_enc(x: torch.Tensor) -> float:
    """Global NVFP4 encode scale, matching NVFP4QuantizerRef exactly:
    S_enc = min(448*6 / global_amax, FLT_MAX); 0/inf global_amax -> 1.0."""
    ga = x.abs().max().to(torch.float32)
    flt_max = torch.finfo(torch.float32).max
    s = torch.minimum(
        (torch.tensor(FP8E4M3_MAX * FP4_E2M1_MAX, dtype=torch.float32, device=x.device) / ga),
        torch.tensor(flt_max, dtype=torch.float32, device=x.device),
    )
    if float(ga) == 0.0 or float(s) == 0.0:
        return 1.0
    return float(s.item())


def quantize_hybrid(x: torch.Tensor, cfg, s_enc_val):
    """Compile + run the hybrid kernel for `cfg`, returning a dict of the
    enabled output torch tensors. Mirrors `quantize_hybrid_cutedsl` but lets
    the test pin an arbitrary cfg (e.g. MXFP8-only for the reference run)."""
    M, N = x.shape
    dev = x.device
    outs = {}
    args_c = [_fake(cfg.DTYPE, (M, N))]
    args_r = [from_dlpack(x, assumed_align=16)]

    def add(name, shape, enabled):
        if enabled:
            t = torch.empty(shape, dtype=torch.uint8, device=dev)
            outs[name] = t
            args_c.append(_fake(cute.Uint8, shape))
            args_r.append(from_dlpack(t, assumed_align=16))
        else:
            args_c.append(None)
            args_r.append(None)

    # Order must match MXFP8QuantizeSmemKernel.__call__:
    #   mO_row, mS_row, mO_col, mS_col, mAmax,
    #   mO_nvfp4_row, mS_nvfp4_row, mO_nvfp4_col, mS_nvfp4_col, s_enc
    add("row_data",  (M, N),               cfg.ROWWISE)
    add("row_scale", (M, N // SCALE_DIM),  cfg.ROWWISE)
    add("col_data",  (M, N),               cfg.COLWISE)
    add("col_scale", (M // SCALE_DIM, N),  cfg.COLWISE)
    args_c.append(None); args_r.append(None)               # mAmax (WITH_AMAX=False)
    add("nv_row_data",  (M, N // 2),              cfg.NVFP4_ROWWISE)
    add("nv_row_scale", (M, N // SCALE_DIM_NVFP4), cfg.NVFP4_ROWWISE)
    add("nv_col_data",  (N, M // 2),              cfg.NVFP4_COLWISE)   # transposed
    add("nv_col_scale", (N, M // SCALE_DIM_NVFP4), cfg.NVFP4_COLWISE)  # transposed

    se = Float32(s_enc_val) if (cfg.NVFP4_ROWWISE or cfg.NVFP4_COLWISE) else None
    args_c.append(se)
    args_r.append(se)

    # Hybrid kernel (11-param __call__); NVFP4 slots are None for the
    # MXFP8-only reference run, which traces the MXFP8 path identically.
    compiled = cute.compile(HybridQuantizeSmemKernel(cfg), *args_c)
    compiled(*args_r)
    torch.cuda.synchronize()
    return outs


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
def nvfp4_reference(x: torch.Tensor):
    """Pure-PyTorch NVFP4 (1x16, non-pow2, no RHT) for both directions."""
    q = NVFP4QuantizerRef(
        dtype=cr_utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=True,
        pow_2_scales=False,
        quant_tile_shape=(1, 16),
        with_rht=False,
        with_random_sign_mask=False,
    )
    return q.quantize(x)


def _assert_eq(tag, got, ref):
    mism = (got != ref).sum().item()
    assert mism == 0, f"{tag}: {mism} / {ref.numel()} byte mismatches"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
SHAPES = [(256, 512), (512, 1024), (128, 256)]


@pytest.mark.parametrize("M,N", SHAPES, ids=lambda v: f"{v}")
@pytest.mark.parametrize("mxfp8_dir", ["row", "col"])
def test_hybrid_bitwise(M, N, mxfp8_dir):
    """MXFP8 in `mxfp8_dir`, NVFP4 in the opposite direction. The NVFP4 output
    must be bitwise-equal to NVFP4QuantizerRef; the MXFP8 output must be
    bitwise-equal to the same kernel run MXFP8-only."""
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    s_enc = compute_s_enc(x)

    rowwise = mxfp8_dir == "row"
    colwise = mxfp8_dir == "col"
    nv_rowwise = not rowwise   # NVFP4 in the other direction
    nv_colwise = not colwise

    cfg_hybrid = MXFP8QuantizeConfig(
        cutlass.BFloat16, "e4m3",
        rowwise=rowwise, colwise=colwise,
        nvfp4_rowwise=nv_rowwise, nvfp4_colwise=nv_colwise,
    )
    cfg_mxfp8_only = MXFP8QuantizeConfig(
        cutlass.BFloat16, "e4m3", rowwise=rowwise, colwise=colwise,
    )

    out = quantize_hybrid(x, cfg_hybrid, s_enc)
    ref_mxfp8 = quantize_hybrid(x, cfg_mxfp8_only, None)
    ref_nv = nvfp4_reference(x)

    # MXFP8 direction: hybrid must equal MXFP8-only (no perturbation).
    if rowwise:
        _assert_eq("MXFP8 row data", out["row_data"], ref_mxfp8["row_data"])
        _assert_eq("MXFP8 row scale", out["row_scale"], ref_mxfp8["row_scale"])
    else:
        _assert_eq("MXFP8 col data", out["col_data"], ref_mxfp8["col_data"])
        _assert_eq("MXFP8 col scale", out["col_scale"], ref_mxfp8["col_scale"])

    # NVFP4 direction: bitwise vs NVFP4QuantizerRef.
    if nv_rowwise:
        _assert_eq("NVFP4 row data", out["nv_row_data"], ref_nv.data)
        _assert_eq("NVFP4 row scale", out["nv_row_scale"], ref_nv.scale.view(torch.uint8))
    else:
        _assert_eq("NVFP4 col data", out["nv_col_data"], ref_nv.data_t)
        _assert_eq("NVFP4 col scale", out["nv_col_scale"], ref_nv.scale_t.view(torch.uint8))


@pytest.mark.parametrize("M,N", [(256, 512)], ids=lambda v: f"{v}")
def test_public_wrapper(M, N):
    """The public `quantize_hybrid_cutedsl` wrapper (default: MXFP8 rowwise +
    NVFP4 colwise) is bitwise-exact against NVFP4QuantizerRef."""
    torch.manual_seed(1)
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    out = quantize_hybrid_cutedsl(x)  # MXFP8 rowwise + NVFP4 colwise
    ref = nvfp4_reference(x)

    assert set(out) >= {"rowwise_data", "rowwise_scale",
                        "nvfp4_colwise_data", "nvfp4_colwise_scale", "s_enc"}
    _assert_eq("NVFP4 col data", out["nvfp4_colwise_data"], ref.data_t)
    _assert_eq("NVFP4 col scale", out["nvfp4_colwise_scale"], ref.scale_t.view(torch.uint8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
