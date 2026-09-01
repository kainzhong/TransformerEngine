# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Cross-backend bit-exactness tests for the CuTeDSL grouped MXFP8 quantize kernel.

Unlike tests/pytorch/mxfp8/test_mxfp8_cutedsl_backend.py, the grouped kernel has no C++
bridge yet, so it cannot be reached through the public dispatch by toggling
nvte_set_cutedsl_quant_backend. Instead the CuTeDSL kernel is invoked directly with the
same marshalling the bridge will use, and its output is compared byte-for-byte against
the CUDA grouped kernel reached through tex.group_quantize.
"""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

from transformer_engine.common.CuTeDSL.cast.mxfp8.group_quantize_mxfp8 import (
    MXFP8GroupQuantizeConfig,
    compile_cutedsl_function_from_cfg,
    NUM_WORKSPACE_SLOTS,
    BYTES_PER_TENSORMAP,
)

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)
pytestmark = pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)

DEV = "cuda"

# The kernel requires each member's rows % 128; the last dim needs % 32 for the
# single-tensor reps and % 128 for varying_last_dim (per-member scale regions pack densely).
GROUP_CASES = [
    ("same_both_dims", [(128, 256), (128, 256)]),
    ("same_both_dims", [(256, 160), (256, 160)]),  # N % 128 != 0 -> ragged column tile
    ("varying_first_dim", [(128, 512), (256, 512)]),
    ("varying_first_dim", [(128, 256), (0, 256), (256, 256)]),  # zero-sized member
    ("varying_last_dim", [(256, 128), (256, 384)]),
    ("varying_last_dim", [(256, 128), (256, 384), (256, 0), (256, 256), (256, 1152)]),
]
GROUP_IDS = [f"{rep}-{'_'.join(f'{m}x{n}' for m, n in shapes)}" for rep, shapes in GROUP_CASES]

IN_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
DTYPE_TO_STR = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}
FP8_DTYPES = [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]
FP8_TO_KEY = {tex.DType.kFloat8E4M3: "fp8_e4m3fn", tex.DType.kFloat8E5M2: "fp8_e5m2"}
FP8_TO_TORCH = {
    tex.DType.kFloat8E4M3: torch.float8_e4m3fn,
    tex.DType.kFloat8E5M2: torch.float8_e5m2,
}
# (block_rows, block_cols): (1,32)=rowwise, (32,1)=colwise, (32,32)=both.
BLOCK_SIZES = [(1, 32), (32, 1), (32, 32)]

get_dtype_id = DTYPE_TO_STR.get
get_fp8_id = {tex.DType.kFloat8E4M3: "e4m3", tex.DType.kFloat8E5M2: "e5m2"}.get
get_block_id = lambda b: f"{b[0]}x{b[1]}"


def roundup(x, m):
    return ((x + m - 1) // m) * m


def build_group(shapes, rep, in_dtype, seed=0):
    """Members concatenated into one flat payload, plus the metadata each rep carries.

    Members are stored as whole contiguous blobs -- for varying_last_dim this is NOT an
    axis-1 concatenation, so the (M, sum N_i) view is bookkeeping only.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    members = [(torch.randn(m, n, device=DEV, generator=g) * 4).to(in_dtype) for m, n in shapes]
    payload = torch.cat([t.reshape(-1) for t in members]).contiguous()
    offsets = [0]
    for m, n in shapes:
        offsets.append(offsets[-1] + m * n)

    if rep == "varying_last_dim":
        logical = (shapes[0][0], sum(n for _, n in shapes))
        first_dims, last_dims = None, [n for _, n in shapes]
    else:
        logical = (sum(m for m, _ in shapes), shapes[0][1])
        first_dims = [m for m, _ in shapes] if rep == "varying_first_dim" else None
        last_dims = None
    return members, payload, offsets, logical, first_dims, last_dims


def run_cuda(payload, logical, num_tensors, first_dims, last_dims, fp8_dtype, rowwise, colwise):
    """Reference: the CUDA grouped kernel through the public dispatch."""
    q = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=rowwise, columnwise=colwise)
    q.optimize_for_gemm = False  # compact (non-swizzled) scales; CuTeDSL has no swizzle path
    to_dev = lambda v: None if v is None else torch.tensor(v, dtype=torch.int64, device=DEV)
    out = tex.group_quantize(
        payload.view(*logical), q, num_tensors, to_dev(first_dims), to_dev(last_dims)
    )
    return out.split_into_quantized_tensors()


def run_cutedsl(payload, logical, shapes, offsets, rep, in_dtype, fp8_dtype, rowwise, colwise):
    """Candidate: the CuTeDSL kernel, marshalled exactly as the C++ bridge will."""
    cfg = MXFP8GroupQuantizeConfig(
        dtype=DTYPE_TO_STR[in_dtype],
        fp8_dtype=FP8_TO_KEY[fp8_dtype],
        rowwise=rowwise,
        colwise=colwise,
        shape_rep=rep,
    )
    fn = compile_cutedsl_function_from_cfg(cfg)

    total = offsets[-1]
    if rep == "varying_last_dim":
        # cols % 128 == 0 for every member, so the scale regions pack densely.
        srow_n = scol_n = total // 32
    else:
        M_total, N = logical
        srow_n = roundup(M_total, 128) * roundup((N + 31) // 32, 4)
        scol_n = roundup(M_total // 32, 4) * roundup(N, 128)

    fp8_t = FP8_TO_TORCH[fp8_dtype]
    o_row = torch.zeros(total, dtype=fp8_t, device=DEV)
    o_col = torch.zeros(total, dtype=fp8_t, device=DEV)
    s_row = torch.zeros(srow_n, dtype=torch.float8_e8m0fnu, device=DEV)
    s_col = torch.zeros(scol_n, dtype=torch.float8_e8m0fnu, device=DEV)
    tmaps = torch.zeros(
        len(shapes), NUM_WORKSPACE_SLOTS, BYTES_PER_TENSORMAP // 8, dtype=torch.int64, device=DEV
    )
    fn(
        payload.view(*logical),
        o_row.view(*logical),
        o_col.view(*logical),
        s_row,
        s_col,
        torch.tensor(offsets, dtype=torch.int64, device=DEV),
        torch.tensor([m for m, _ in shapes], dtype=torch.int64, device=DEV),
        torch.tensor([n for _, n in shapes], dtype=torch.int64, device=DEV),
        tmaps,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return o_row, o_col, s_row, s_col


def member_views(i, shapes, offsets, logical, rep, o_row, o_col, s_row, s_col, row_base):
    """Slice member i's meaningful (unpadded) region out of the flat CuTeDSL buffers."""
    mi, ni = shapes[i]
    data_row = o_row.view(torch.uint8)[offsets[i] : offsets[i + 1]].view(mi, ni)
    data_col = o_col.view(torch.uint8)[offsets[i] : offsets[i + 1]].view(mi, ni)
    sr8, sc8 = s_row.view(torch.uint8), s_col.view(torch.uint8)
    if rep == "varying_last_dim":
        lo, hi = offsets[i] // 32, offsets[i + 1] // 32
        scale_row = sr8[lo:hi].view(mi, ni // 32)
        scale_col = sc8[lo:hi].view(mi // 32, ni)
    else:
        m_total, n = logical
        scale_row = sr8.view(roundup(m_total, 128), -1)[row_base : row_base + mi, : (ni + 31) // 32]
        scale_col = sc8.view(-1, roundup(n, 128))[row_base // 32 : (row_base + mi) // 32, :ni]
    return data_row, data_col, scale_row, scale_col


def run_test_case(rep, shapes, in_dtype, fp8_dtype, block_size):
    rowwise = block_size[1] != 1
    colwise = block_size[0] != 1
    members, payload, offsets, logical, first_dims, last_dims = build_group(shapes, rep, in_dtype)

    ref = run_cuda(
        payload, logical, len(shapes), first_dims, last_dims, fp8_dtype, rowwise, colwise
    )
    o_row, o_col, s_row, s_col = run_cutedsl(
        payload, logical, shapes, offsets, rep, in_dtype, fp8_dtype, rowwise, colwise
    )

    row_base = 0
    for i, (mi, ni) in enumerate(shapes):
        if mi == 0 or ni == 0:
            # A zero-sized member is a valid group entry; the kernel skips it and the
            # buffer is never written, so there is nothing to compare.
            row_base += mi
            continue
        data_row, data_col, scale_row, scale_col = member_views(
            i, shapes, offsets, logical, rep, o_row, o_col, s_row, s_col, row_base
        )
        tag = f"{rep}/member {i} ({mi}x{ni})/{DTYPE_TO_STR[in_dtype]}/{get_fp8_id(fp8_dtype)}"
        # The reference scale tensors carry the [128,4] / [4,128] alignment padding. CUDA
        # zeroes that padding and the CuTeDSL helpers skip it, so compare only the
        # meaningful region -- same convention as the single-tensor backend test.
        if rowwise:
            assert torch.equal(
                data_row, ref[i]._rowwise_data.view(torch.uint8)
            ), f"{tag}: rowwise data differ between backends"
            assert torch.equal(
                scale_row, ref[i]._rowwise_scale_inv.view(torch.uint8)[:mi, : (ni + 31) // 32]
            ), f"{tag}: rowwise scales differ between backends"
        if colwise:
            assert torch.equal(
                data_col, ref[i]._columnwise_data.view(torch.uint8)
            ), f"{tag}: colwise data differ between backends"
            assert torch.equal(
                scale_col, ref[i]._columnwise_scale_inv.view(torch.uint8)[: mi // 32, :ni]
            ), f"{tag}: colwise scales differ between backends"
        row_base += mi


# Every shape representation and member layout the CuTeDSL grouped kernel supports.
@pytest.mark.parametrize("rep,shapes", GROUP_CASES, ids=GROUP_IDS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES, ids=get_block_id)
def test_group_cast_only(block_size, rep, shapes):
    run_test_case(rep, shapes, torch.bfloat16, tex.DType.kFloat8E4M3, block_size)


# Input and FP8 dtype coverage on one representative group per representation.
@pytest.mark.parametrize("in_dtype", IN_DTYPES, ids=get_dtype_id)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES, ids=get_fp8_id)
@pytest.mark.parametrize("rep,shapes", GROUP_CASES[::2], ids=GROUP_IDS[::2])
def test_group_dtypes(rep, shapes, fp8_dtype, in_dtype):
    run_test_case(rep, shapes, in_dtype, fp8_dtype, (32, 32))


# Groups large enough that a CTA processes several chunks (varying_last_dim) or the
# direct-mapper grid exceeds one wave.
@pytest.mark.parametrize(
    "rep,shapes",
    [
        ("same_both_dims", [(8192, 4096), (8192, 4096)]),
        ("varying_last_dim", [(4096, 4096), (4096, 8192), (4096, 4096)]),
    ],
    ids=["same_both_dims-large", "varying_last_dim-large"],
)
def test_group_large(rep, shapes):
    run_test_case(rep, shapes, torch.bfloat16, tex.DType.kFloat8E4M3, (32, 32))
