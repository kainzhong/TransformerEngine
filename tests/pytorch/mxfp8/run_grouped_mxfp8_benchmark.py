#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""One-shot driver for the GROUPED MXFP8 quantize benchmark.

Same structure as run_mxfp8_benchmark.py, but for the grouped (multi-tensor)
kernel. One difference forces this to be a single self-contained file rather than
a driver + a separate bench: the grouped CuTeDSL kernel has NO C++ bridge yet, so
the backend cannot be latched with NVTE_ENABLE_CUTEDSL_QUANT_BACKEND. Instead:

    backend:  cpp = CUDA grouped kernel   via tex.group_quantize
              dsl = CuTeDSL grouped kernel via the compiled cute function, called
                    with exactly the marshalling the C++ bridge will use
                    (see tests/pytorch/mxfp8/test_mxfp8_group_cutedsl_backend.py)
    mode:     GPU = kernel time, cold L2   (nsys NVTX Range Kernel Summary; the
                                            whole matrix in ONE nsys run per
                                            backend, per-workload via NVTX; needs nsys)
              CPU = host dispatch time     (in-process wall clock, no sync/no flush)

So this script is both the driver (default) and the worker (--worker, one backend
per subprocess so a CuTeDSL JIT/compile never lands inside the cpp measurement).

A workload is a GROUP: a shape representation + a member shape list. Both
backends see the identical payload, quantizer settings (compact/non-swizzled
scales) and preallocated destinations, so only the kernel differs. Scope matches
the CuTeDSL kernel: cast-only, compact scales, rowwise and/or colwise; the
varying_both_dims rep is not supported and is not benchmarked.

Usage:
    python run_grouped_mxfp8_benchmark.py                       # curated default
    python run_grouped_mxfp8_benchmark.py --preset moe --modes gpu
    python run_grouped_mxfp8_benchmark.py --groups moe8_vfd --directions both
    python run_grouped_mxfp8_benchmark.py --backends dsl --list-groups
    python run_grouped_mxfp8_benchmark.py --groups 'vfd:1024x4096,2048x4096'
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SELF = Path(__file__).resolve()
# Repo root (…/tests/pytorch/mxfp8/run_grouped_mxfp8_benchmark.py -> 3 levels up).
# The worker is launched as a script, so its own dir lands on sys.path and
# `import transformer_engine` would resolve to the INSTALLED package — not this
# checkout — meaning `dsl` would silently benchmark stale installed kernels.
# Putting the repo root on PYTHONPATH forces the local TE.
_REPO_ROOT = SELF.parents[3]

# Kernel-summary rows whose name matches any of these are the L2-evict op or an
# allocator fill, not the quantize — excluded from the measured GPU time.
_EVICT_NAME_PATTERNS = ("memset", "memcpy", "fill", "elementwise_kernel",
                        "vectorized_elementwise")

# Shape representations (mirror ShapeRepresentation in common/utils.cuh). Short
# aliases are what --groups accepts.
_REPS = {"sbd": "same_both_dims", "vfd": "varying_first_dim", "vld": "varying_last_dim"}
_REP_ALIAS = {v: k for k, v in _REPS.items()}

def _ragged(total, n, other, spread=3.0, vary="rows"):
    """n member shapes with a ragged, MoE-routing-like split of `total`.

    The varying extent is a linear ramp with max/min ~= `spread`, every part a multiple
    of 128 (the kernel's chunk dim), summing EXACTLY to `total` so the group's logical
    shape is exactly the one the preset advertises. `vary="rows"` -> (part, other) for
    varying_first_dim; `vary="cols"` -> (other, part) for varying_last_dim.
    """
    w = [1.0 + (spread - 1.0) * (i / max(n - 1, 1)) for i in range(n)]
    tot_w = sum(w)
    parts = [max(128, int(round(total * x / tot_w / 128)) * 128) for x in w]
    parts[parts.index(max(parts))] += total - sum(parts)  # remainder is a multiple of 128
    assert all(p >= 128 and p % 128 == 0 for p in parts) and sum(parts) == total
    return [(p, other) if vary == "rows" else (other, p) for p in parts]


# ---------------------------------------------------------------------------
# Group presets: name -> (shape_rep, [(rows, cols), ...]).
#
# Constraints the grouped kernel imposes (see the CuTeDSL kernel docstring and
# the cross-backend test): every member's rows % 128 == 0, cols % 32 == 0, and
# for varying_last_dim cols % 128 == 0 (per-member scale regions pack densely).
# same_both_dims  -> every member identical
# varying_first_dim -> members share cols
# varying_last_dim  -> members share rows
# ---------------------------------------------------------------------------
GROUP_PRESETS = {
    # --- quick smoke sizes ---
    "tiny_sbd":  ("same_both_dims",    [(128, 256)] * 2),
    "tiny_vfd":  ("varying_first_dim", [(128, 512), (256, 512)]),
    "tiny_vld":  ("varying_last_dim",  [(256, 128), (256, 384)]),

    # --- MoE-shaped: 8 experts, hidden 4096 (uniform vs ragged token counts) ---
    "moe8_sbd":  ("same_both_dims",    [(2048, 4096)] * 8),
    "moe8_vfd":  ("varying_first_dim",
                  [(2048, 4096), (1024, 4096), (4096, 4096), (512, 4096),
                   (1536, 4096), (3072, 4096), (768, 4096), (2560, 4096)]),
    # Heavily imbalanced expert routing — stresses the persistent scheduler.
    "moe8_skew": ("varying_first_dim",
                  [(128, 4096), (128, 4096), (128, 4096), (128, 4096),
                   (128, 4096), (128, 4096), (128, 4096), (14336, 4096)]),
    # --- 4 experts, Llama-3 MLP intermediate widths (varying last dim) ---
    "moe4_vld":  ("varying_last_dim",
                  [(4096, 4096), (4096, 8192), (4096, 14336), (4096, 2048)]),

    # --- larger group counts (descriptor prologue / grid pressure) ---
    "moe32_sbd": ("same_both_dims",    [(1024, 4096)] * 32),
    "moe64_vfd": ("varying_first_dim", [(512 * (1 + i % 4), 2048) for i in range(64)]),

    # --- MoE layers from real model configs (the default set) ---------------
    # Grouped MXFP8 quantize is what feeds a grouped GEMM in an MoE layer, so the
    # member list is one expert's token block: (tokens_routed_to_expert, hidden) for
    # fc1, (tokens, moe_intermediate) for fc2. Token counts are ragged, as real
    # top-k routing produces; the sbd variants are the perfectly-balanced ideal.
    #
    # 32768 x 7168 in all three shape representations (DeepSeek-V3 hidden = 7168,
    # 32768 routed tokens). Identical logical shape and byte count in each, so the
    # three reps -- and the two code paths they exercise -- are directly comparable.
    "moe32k_sbd": ("same_both_dims",    [(4096, 7168)] * 8),
    "moe32k_vfd": ("varying_first_dim", _ragged(32768, 8, 7168)),
    "moe32k_vld": ("varying_last_dim",  _ragged(7168, 8, 32768, vary="cols")),

    # DeepSeek-V3 / R1: hidden 7168, moe intermediate 2048, 256 routed experts (top-8).
    # 32 experts' worth of a 32768-token batch.
    "dsv3_fc1":  ("varying_first_dim", _ragged(32768, 32, 7168)),
    "dsv3_fc2":  ("varying_first_dim", _ragged(32768, 32, 2048)),
    # Mixtral 8x7B: hidden 4096, intermediate 14336, 8 experts (top-2).
    "mixtral_fc1": ("varying_first_dim", _ragged(16384, 8, 4096)),
    "mixtral_fc2": ("varying_first_dim", _ragged(16384, 8, 14336)),
    # Qwen3-235B-A22B: hidden 4096, moe intermediate 1536, 128 experts (top-8).
    "qwen3_fc1": ("varying_first_dim", _ragged(32768, 64, 4096)),
    # Llama-4 Maverick: hidden 5120, intermediate 8192.
    "llama4_fc1": ("varying_first_dim", _ragged(16384, 16, 5120)),

    # --- rep comparison: three matched triples (sbd / vfd / vld) ---
    # Within a triple the TOTAL element count is identical, so the three shape
    # representations are directly comparable: only the member layout (and hence
    # the code path -- sbd/vfd are is_single_tensor, vld is not) differs.
    #   small : 29.4M elts, 4 members    med : 117.4M elts, 4 members
    #   many  : 117.4M elts, 16 members
    "rep_small_sbd": ("same_both_dims",    [(1024, 7168)] * 4),
    "rep_small_vfd": ("varying_first_dim", [(256, 7168), (768, 7168),
                                            (1280, 7168), (1792, 7168)]),
    "rep_small_vld": ("varying_last_dim",  [(1024, 4096), (1024, 8192),
                                            (1024, 14336), (1024, 2048)]),
    "rep_med_sbd":   ("same_both_dims",    [(4096, 7168)] * 4),
    "rep_med_vfd":   ("varying_first_dim", [(1024, 7168), (3072, 7168),
                                            (5120, 7168), (7168, 7168)]),
    "rep_med_vld":   ("varying_last_dim",  [(4096, 4096), (4096, 8192),
                                            (4096, 14336), (4096, 2048)]),
    "rep_many_sbd":  ("same_both_dims",    [(1024, 7168)] * 16),
    "rep_many_vfd":  ("varying_first_dim",
                      [(r, 7168) for _ in range(4) for r in (384, 768, 1152, 1792)]),
    "rep_many_vld":  ("varying_last_dim",
                      [(1024, c) for _ in range(4) for c in (4096, 8192, 14336, 2048)]),

    # --- big single-rep groups (bandwidth ceiling) ---
    "big_sbd":   ("same_both_dims",    [(8192, 4096)] * 4),
    "big_vld":   ("varying_last_dim",  [(4096, 4096), (4096, 8192), (4096, 4096)]),
}

# Curated default: the 32768x7168 triple (all three shape representations at an
# identical logical shape) plus four real MoE layers. `--preset moe` adds more.
_DEFAULT_GROUPS = ("moe32k_sbd,moe32k_vfd,moe32k_vld,"
                   "dsv3_fc1,dsv3_fc2,mixtral_fc1,qwen3_fc1")

_ALL_DTYPES = ["bf16", "fp16", "fp32"]
_ALL_FP8 = ["e4m3", "e5m2"]

# name -> comma list of preset names.
PRESETS = {
    "tiny":    "tiny_sbd,tiny_vfd,tiny_vld",
    "moe":     "moe8_sbd,moe8_vfd,moe8_skew,moe4_vld",
    # Every real-model MoE layer preset.
    "llm":     ("moe32k_sbd,moe32k_vfd,moe32k_vld,dsv3_fc1,dsv3_fc2,"
                "mixtral_fc1,mixtral_fc2,qwen3_fc1,llama4_fc1"),
    "ngroups": "moe8_sbd,moe32_sbd,moe64_vfd",
    "large":   "big_sbd,big_vld,moe8_skew",
    # Matched sbd/vfd/vld triples at equal total bytes — the rep comparison.
    "reps":    ("rep_small_sbd,rep_small_vfd,rep_small_vld,"
                "rep_med_sbd,rep_med_vfd,rep_med_vld,"
                "rep_many_sbd,rep_many_vfd,rep_many_vld"),
    "all":     ",".join(GROUP_PRESETS),
}


def split_group_specs(spec_str):
    """Split a --groups value into individual group specs.

    ';' always separates groups; ',' separates them only inside a chunk that is
    NOT an inline 'rep:MxN,MxN' spec (whose member list is itself comma-separated).
    So 'moe8_sbd,moe4_vld;vld:4096x7168,4096x7168' -> 3 groups.
    """
    out = []
    for chunk in spec_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            out.append(chunk)
        else:
            out += [t.strip() for t in chunk.split(",") if t.strip()]
    return out


def parse_group_spec(spec):
    """Resolve one --groups token to (label, rep, shapes).

    Either a preset name, or an inline 'rep:MxN,MxN,...' where rep is one of
    sbd/vfd/vld (or the full same_both_dims/varying_first_dim/varying_last_dim).
    """
    if spec in GROUP_PRESETS:
        rep, shapes = GROUP_PRESETS[spec]
        return spec, rep, shapes
    if ":" not in spec:
        raise ValueError(
            f"unknown group {spec!r}; expected a preset (see --list-groups) or "
            "'rep:MxN,MxN,...' with rep in sbd|vfd|vld")
    rep_tok, shape_tok = spec.split(":", 1)
    rep = _REPS.get(rep_tok, rep_tok)
    if rep not in _REPS.values():
        raise ValueError(f"unknown shape representation {rep_tok!r}")
    shapes = []
    for t in shape_tok.split(","):
        t = t.strip().lower()
        if t:
            m, n = t.split("x")
            shapes.append((int(m), int(n)))
    if not shapes:
        raise ValueError(f"group {spec!r} has no member shapes")
    label = f"{_REP_ALIAS[rep]}{len(shapes)}x_custom"
    return label, rep, shapes


def group_logical_shape(rep, shapes):
    """The 2D logical view the group is passed through (bookkeeping for vld)."""
    if rep == "varying_last_dim":
        return (shapes[0][0], sum(n for _, n in shapes))
    return (sum(m for m, _ in shapes), shapes[0][1])


# ===========================================================================
# Worker: measures ONE backend in ONE timing mode. Torch is imported here only.
# ===========================================================================

def _worker_main(args):
    import time

    import torch
    import torch.cuda.nvtx as nvtx

    import transformer_engine.pytorch as te  # must precede transformer_engine_torch
    import transformer_engine_torch as tex
    from transformer_engine.pytorch import MXFP8Quantizer

    DEV = "cuda"
    _TORCH_IN = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    _FP8 = {"e4m3": tex.DType.kFloat8E4M3, "e5m2": tex.DType.kFloat8E5M2}
    _FP8_TORCH = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
    _FP8_KEY = {"e4m3": "fp8_e4m3fn", "e5m2": "fp8_e5m2"}
    _DIR_RCW = {"row": (True, False), "col": (False, True), "both": (True, True)}

    backend = args.backend

    def roundup(x, m):
        return ((x + m - 1) // m) * m

    # ---- payload ---------------------------------------------------------
    def build_group(shapes, rep, in_dtype, seed=0):
        """Members concatenated into ONE flat payload + the metadata each rep carries.

        Members are whole contiguous blobs — for varying_last_dim this is NOT an
        axis-1 concat, so the (M, sum N_i) view is bookkeeping only. Same layout
        as test_mxfp8_group_cutedsl_backend.build_group.
        """
        g = torch.Generator(device=DEV).manual_seed(seed)
        total = sum(m * n for m, n in shapes)
        payload = (torch.randn(total, device=DEV, generator=g) * 4).to(in_dtype).contiguous()
        offsets = [0]
        for m, n in shapes:
            offsets.append(offsets[-1] + m * n)
        logical = group_logical_shape(rep, shapes)
        first_dims = [m for m, _ in shapes] if rep == "varying_first_dim" else None
        last_dims = [n for _, n in shapes] if rep == "varying_last_dim" else None
        return payload, offsets, logical, first_dims, last_dims

    def workload_bytes(shapes, in_bytes_per_elt, rowwise, colwise):
        """HBM traffic lower bound: input read + FP8 out + e8m0 scales, per member."""
        b = 0
        for m, n in shapes:
            b += m * n * in_bytes_per_elt
            if rowwise:
                b += m * n + m * (n // 32)
            if colwise:
                b += m * n + (m // 32) * n
        return b

    # ---- cpp: the CUDA grouped kernel through the public dispatch ---------
    def make_cpp_fn(payload, logical, shapes, offsets, first_dims, last_dims,
                    fp8, rowwise, colwise):
        q = MXFP8Quantizer(fp8_dtype=_FP8[fp8], rowwise=rowwise, columnwise=colwise)
        # Compact (non-swizzled) scales: the CuTeDSL kernel has no swizzle path,
        # so this is what makes the two backends comparable.
        q.optimize_for_gemm = False
        to_dev = lambda v: None if v is None else torch.tensor(v, dtype=torch.int64, device=DEV)
        fd, ld = to_dev(first_dims), to_dev(last_dims)
        x = payload.view(*logical)
        n_t = len(shapes)
        # Hand the CSR offsets in precomputed, exactly as the dsl arm does. Without
        # this, group_quantize derives them per call via build_grouped_tensor_offsets
        # -> nvte_splits_to_offsets, a second kernel INSIDE the timed NVTX range worth
        # a size-independent ~2.4 us (it is O(num_tensors)). That made every rep which
        # passes first_dims/last_dims look ~2.4 us slower than the CuTeDSL arm, which
        # builds its offsets tensor once outside fn(). Offset resolution is upstream of
        # the backend choice in production (resolve_grouped_tensor_offsets, shared by
        # both paths), so hoisting it out of both arms is what isolates the kernel.
        # same_both_dims passes neither first_dims nor last_dims, so it never derived
        # offsets at all -- leave it None there rather than changing what it measures.
        off = (torch.tensor(offsets, dtype=torch.int64, device=DEV)
               if (fd is not None or ld is not None) else None)
        # Allocate the destination ONCE and reuse it (`output=`), so the timed loop
        # measures the kernel, not the allocator — matching the dsl path, which
        # writes into preallocated buffers. Destination reuse is only implemented
        # for uniform member shapes (cast.cpp: "output reuse currently requires
        # uniform tensor shapes"), so the ragged reps fall back to allocating each
        # call. That only shifts CPU-mode (host dispatch) numbers; GPU mode reads
        # kernel time, and the allocation issues no kernel.
        out = tex.group_quantize(x, q, n_t, fd, ld, off)
        try:
            tex.group_quantize(x, q, n_t, fd, ld, off, output=out)
            reuse = True
        except RuntimeError:
            reuse = False

        if reuse:
            def fn():
                return tex.group_quantize(x, q, n_t, fd, ld, off, output=out)
        else:
            def fn():
                return tex.group_quantize(x, q, n_t, fd, ld, off)

        return fn

    # ---- dsl: the CuTeDSL grouped kernel, marshalled as the bridge will ----
    _dsl_cache = {}

    def _cutedsl_compat():
        """Older CuTeDSL wheels expose TensorMapManager from `cutlass.utils`, newer
        ones from `cutlass.tensor_utils` (what the kernel imports). Alias the new
        path onto the old module when it is missing so the bench runs on both; a
        no-op where `cutlass.tensor_utils` already exists."""
        import importlib
        import sys as _sys

        try:
            importlib.import_module("cutlass.tensor_utils")
        except ModuleNotFoundError:
            cutlass_utils = importlib.import_module("cutlass.utils")
            for sym in ("TensorMapManager", "TensorMapUpdateMode"):
                if not hasattr(cutlass_utils, sym):
                    raise
            _sys.modules["cutlass.tensor_utils"] = cutlass_utils

    def dsl_fn_for_cfg(in_dtype, fp8, rowwise, colwise, rep):
        _cutedsl_compat()
        from transformer_engine.common.CuTeDSL.cast.mxfp8.group_quantize_mxfp8 import (
            MXFP8GroupQuantizeConfig,
            compile_cutedsl_function_from_cfg,
        )

        key = (in_dtype, fp8, rowwise, colwise, rep)
        if key not in _dsl_cache:
            cfg = MXFP8GroupQuantizeConfig(
                dtype=in_dtype, fp8_dtype=_FP8_KEY[fp8],
                rowwise=rowwise, colwise=colwise, shape_rep=rep,
            )
            # Compiled with symbolic extents, so ONE compile serves every shape.
            _dsl_cache[key] = compile_cutedsl_function_from_cfg(cfg)
        return _dsl_cache[key]

    def make_dsl_fn(payload, logical, shapes, offsets, rep, in_dtype, fp8, rowwise, colwise):
        _cutedsl_compat()
        from transformer_engine.common.CuTeDSL.cast.mxfp8.group_quantize_mxfp8 import (
            NUM_WORKSPACE_SLOTS,
            BYTES_PER_TENSORMAP,
        )

        fn_c = dsl_fn_for_cfg(in_dtype, fp8, rowwise, colwise, rep)
        total = offsets[-1]
        if rep == "varying_last_dim":
            srow_n = scol_n = total // 32
        else:
            M_total, N = logical
            srow_n = roundup(M_total, 128) * roundup((N + 31) // 32, 4)
            scol_n = roundup(M_total // 32, 4) * roundup(N, 128)

        fp8_t = _FP8_TORCH[fp8]
        o_row = torch.zeros(total, dtype=fp8_t, device=DEV)
        o_col = torch.zeros(total, dtype=fp8_t, device=DEV)
        s_row = torch.zeros(srow_n, dtype=torch.float8_e8m0fnu, device=DEV)
        s_col = torch.zeros(scol_n, dtype=torch.float8_e8m0fnu, device=DEV)
        tmaps = torch.zeros(len(shapes), NUM_WORKSPACE_SLOTS, BYTES_PER_TENSORMAP // 8,
                            dtype=torch.int64, device=DEV)
        x = payload.view(*logical)
        xr = o_row.view(*logical)
        xc = o_col.view(*logical)
        off_t = torch.tensor(offsets, dtype=torch.int64, device=DEV)
        fd_t = torch.tensor([m for m, _ in shapes], dtype=torch.int64, device=DEV)
        ld_t = torch.tensor([n for _, n in shapes], dtype=torch.int64, device=DEV)

        def fn():
            fn_c(x, xr, xc, s_row, s_col, off_t, fd_t, ld_t, tmaps,
                 torch.cuda.current_stream().cuda_stream)

        return fn

    # ---- L2 evict buffer (shared, allocated lazily) -----------------------
    evict_buf = {}

    def l2_evict_buf():
        if "b" not in evict_buf:
            # 256 MB f32 — covers B200/GB200's L2 with headroom.
            evict_buf["b"] = torch.empty(256 * 1024 * 1024 // 4,
                                         dtype=torch.float32, device=DEV)
        return evict_buf["b"]

    # ---- axes ------------------------------------------------------------
    groups = [parse_group_spec(s) for s in split_group_specs(args.groups)]
    in_dtypes = [d.strip() for d in args.in_dtypes.split(",") if d.strip()]
    fp8s = [d.strip() for d in args.fp8s.split(",") if d.strip()]
    dirs = [d.strip() for d in args.directions.split(",") if d.strip()]

    print(f"Backend: {backend}  mode: {'gpu-nsys' if args.gpu_nsys else 'cpu (host dispatch)'}")
    print(f"  groups: {[g[0] for g in groups]}  dirs: {dirs}  "
          f"in_dtypes: {in_dtypes}  fp8: {fp8s}")
    print(f"  warmup={args.warmup} iters={args.iters}", flush=True)

    results = []
    for label, rep, shapes in groups:
        for in_dtype in in_dtypes:
            payload, offsets, logical, first_dims, last_dims = build_group(
                shapes, rep, _TORCH_IN[in_dtype])
            M, N = logical
            elt = payload.element_size()
            for fp8 in fp8s:
                for d in dirs:
                    rowwise, colwise = _DIR_RCW[d]
                    tag = f"{label}_{_REP_ALIAS[rep]}{len(shapes)}t_{in_dtype}_{fp8}"
                    total_bytes = workload_bytes(shapes, elt, rowwise, colwise)
                    if backend == "cpp":
                        fn = make_cpp_fn(payload, logical, shapes, offsets,
                                         first_dims, last_dims, fp8, rowwise, colwise)
                    else:
                        fn = make_dsl_fn(payload, logical, shapes, offsets, rep,
                                         in_dtype, fp8, rowwise, colwise)

                    if args.gpu_nsys:
                        # The driver profiles the WHOLE matrix in one nsys run per
                        # backend and attributes per-workload kernel time via NVTX
                        # ranges (nsys nvtx_kern_sum): the kernel NAME encodes the
                        # config but not the group, so each workload is wrapped in a
                        # same-named QBENCH range that carries it. Warmup and the
                        # L2-evict sit OUTSIDE the range and are ignored. Emit the
                        # byte count so the driver can compute GB/s.
                        print(f"NSYS_BYTES backend={backend} tag={tag} M={M} N={N} "
                              f"dir={d} bytes={total_bytes} iters={args.iters}", flush=True)
                        for _ in range(max(args.warmup, 1)):  # incl. the CuTeDSL JIT
                            fn()
                        torch.cuda.synchronize()
                        evict = l2_evict_buf()
                        rng = f"QBENCH|{tag}|{M}x{N}|{d}"
                        for _ in range(args.iters):
                            evict.zero_()            # cold L2, OUTSIDE the range
                            torch.cuda.synchronize()
                            nvtx.range_push(rng)     # same name every iter ->
                            fn()                     # nvtx_kern_sum aggregates
                            torch.cuda.synchronize()
                            nvtx.range_pop()
                        continue

                    # CPU mode: warm cache, tight launch loop, NO sync, NO flush —
                    # host dispatch cost. GB/s is meaningless here, so it is not
                    # reported for this mode by the driver.
                    for _ in range(args.warmup):
                        fn()
                    torch.cuda.synchronize()
                    nvtx.range_push(f"{backend}_{tag}_{M}x{N}_{d}")
                    t0 = time.perf_counter_ns()
                    for _ in range(args.iters):
                        fn()
                    t1 = time.perf_counter_ns()
                    nvtx.range_pop()
                    us = (t1 - t0) / 1e3 / args.iters
                    gbps = total_bytes / (us * 1e-6) / 1e9
                    results.append((backend, tag, M, N, d, us, gbps, total_bytes))
                    print(f"  {tag:>40}  {f'{M}x{N}':>13}  {d:>4}  {us:9.2f} us", flush=True)

    if args.csv and results:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["backend", "tag", "M", "N", "dir", "us", "gbps", "bytes"])
            for r in results:
                w.writerow([r[0], r[1], r[2], r[3], r[4], f"{r[5]:.3f}", f"{r[6]:.2f}", r[7]])
    return 0


# ===========================================================================
# Driver
# ===========================================================================

def _detect_cute_dsl_arch():
    """sm_<major><minor>[a] for the current device (CuTeDSL compile target)."""
    try:
        import torch

        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}{'a' if major >= 9 else ''}"
    except Exception:
        return None


def _worker_env(backend):
    """Process env for a worker run (CuTeDSL arch + local-TE import path)."""
    env = dict(os.environ)
    if backend == "dsl" and "CUTE_DSL_ARCH" not in env:
        arch = _detect_cute_dsl_arch()
        if arch:
            env["CUTE_DSL_ARCH"] = arch
    # Force the worker to import THIS checkout's transformer_engine, not the
    # installed package — otherwise `dsl` benchmarks stale installed kernels.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + existing if existing else "")
    return env


def _worker_cmd(backend, args, extra):
    return [sys.executable, str(SELF), "--worker", "--backend", backend,
            "--groups", args.groups, "--directions", args.directions,
            "--in-dtypes", args.in_dtypes, "--fp8s", args.fp8s,
            "--warmup", str(args.warmup), "--iters", str(args.iters)] + extra


def _run_cpu(backend, args):
    """CPU mode: worker loops the matrix in-process, times host dispatch, writes
    CSV. Returns {(tag, M, N, dir): (us, gbps)}."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    cmd = _worker_cmd(backend, args, ["--csv", csv_path])
    print(f"[run] backend={backend:3s} mode=cpu", file=sys.stderr)
    try:
        proc = subprocess.run(cmd, env=_worker_env(backend), stdout=subprocess.DEVNULL)
        if proc.returncode != 0:
            print(f"[warn] backend={backend} mode=cpu exited {proc.returncode}; skipping.",
                  file=sys.stderr)
            return {}
        rows = {}
        if os.path.getsize(csv_path) == 0:
            return {}
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                rows[(r["tag"], int(r["M"]), int(r["N"]), r["dir"])] = (
                    float(r["us"]), float(r["gbps"]))
        return rows
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


def _parse_nsys_bytes_all(stdout):
    """All NSYS_BYTES lines -> {(tag, M, N, dir): bytes}. The worker emits one per
    workload before its timed loop."""
    out = {}
    for line in stdout.splitlines():
        if line.startswith("NSYS_BYTES"):
            kv = dict(tok.split("=", 1) for tok in line.split()[1:] if "=" in tok)
            try:
                out[(kv["tag"], int(kv["M"]), int(kv["N"]), kv["dir"])] = int(kv["bytes"])
            except (KeyError, ValueError):
                continue
    return out


def _parse_nvtx_kern_sum(stats_csv, iters):
    """Per-workload per-iter kernel time (ns) from `nsys stats --report
    nvtx_kern_sum --format csv`. Each row is (NVTX range, kernel) with the real
    CUPTI 'Total Time (ns)'. We sum the kernel Total over each QBENCH range and
    divide by its range-instance count (== iters) -> per-iter kernel time,
    bucketed by workload. For the grouped kernel this correctly folds the CuTeDSL
    descriptor-prologue kernel into the total. The L2-evict sits in the blank
    range (outside QBENCH) and is skipped."""
    lines = stats_csv.splitlines()
    hdr = next((i for i, ln in enumerate(lines)
                if "NVTX Range" in ln and "Total Time" in ln), None)
    if hdr is None:
        return {}
    reader = csv.DictReader(lines[hdr:])
    fields = reader.fieldnames or []
    rng_col = next((c for c in fields if c.strip() == "NVTX Range"), None)
    tot_col = next((c for c in fields if "Total Time" in c), None)
    inst_col = next((c for c in fields if c.strip() == "NVTX Inst"), None)
    name_col = next((c for c in fields if "Kernel Name" in c), None)
    if not (rng_col and tot_col and inst_col):
        return {}
    acc = {}  # range key -> [summed kernel Total ns, range instances]
    for row in reader:
        name = (row.get(rng_col) or "").lstrip(":").strip()
        if not name.startswith("QBENCH|"):
            continue
        kname = (row.get(name_col) or "").lower() if name_col else ""
        if any(p in kname for p in _EVICT_NAME_PATTERNS):
            continue  # defensive; evict is in the blank range anyway
        parts = name.split("|")            # QBENCH | tag | MxN | dir
        if len(parts) != 4:
            continue
        _, tag, mxn, d = parts
        try:
            M, N = (int(v) for v in mxn.lower().split("x"))
            tot = float((row[tot_col] or "0").replace(",", ""))
            inst = int(float((row[inst_col] or "0").replace(",", "")))
        except (TypeError, ValueError):
            continue
        cur = acc.setdefault((tag, M, N, d), [0.0, inst])
        cur[0] += tot
        cur[1] = inst or cur[1]
    return {k: (tot / (inst or iters)) for k, (tot, inst) in acc.items() if tot > 0}


def _run_gpu_nsys_backend(nsys, backend, args, out_dir=None):
    """Profile the WHOLE matrix for one backend in a SINGLE nsys run; attribute
    per-workload kernel time via NVTX ranges. Returns {(tag,M,N,dir): (us, gbps)}.

    One process (not one per workload) because the kernel NAME can't separate
    groups — the QBENCH NVTX range carries the group tag and logical shape.
    Whole-process profile flushes reliably at exit. CuTeDSL JIT-compiles each
    distinct config on first use; that all happens here, during each workload's
    warmup, outside the measured ranges."""
    tmp = None if out_dir else tempfile.TemporaryDirectory()
    rep = (os.path.join(out_dir, f"{backend}_grouped_matrix") if out_dir
           else os.path.join(tmp.name, "rep"))
    try:
        cmd = ([nsys, "profile", "-o", rep, "-f", "true", "--resolve-symbols=false"]
               + _worker_cmd(backend, args, ["--gpu-nsys"]))
        print(f"[run] backend={backend} nsys (full matrix in 1 process): "
              f"groups={args.groups} dirs={args.directions}", file=sys.stderr)
        proc = subprocess.run(cmd, env=_worker_env(backend), capture_output=True, text=True)
        if out_dir:
            with open(rep + ".log", "w") as fh:
                fh.write(f"$ {' '.join(cmd)}\n[exit {proc.returncode}]\n"
                         f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
        if proc.returncode != 0:
            print(f"[warn] nsys profile failed ({proc.returncode}) for backend={backend}; "
                  f"skipping.\n{proc.stderr[-800:]}", file=sys.stderr)
            return {}
        bytes_map = _parse_nsys_bytes_all(proc.stdout)
        # --force-export=true: always re-derive the SQLite from the freshly
        # captured .nsys-rep. Otherwise `nsys stats` reuses a stale .sqlite left
        # next to a reused output path, silently reporting old data.
        stats = subprocess.run(
            [nsys, "stats", "--force-export=true", "--report", "nvtx_kern_sum",
             "--format", "csv", rep + ".nsys-rep"], capture_output=True, text=True)
        if out_dir:
            with open(rep + ".nvtx_kern_sum.csv", "w") as fh:
                fh.write(stats.stdout)
        if stats.returncode != 0:
            print(f"[warn] nsys stats failed for backend={backend}; skipping."
                  f"\n{stats.stderr[-500:]}", file=sys.stderr)
            return {}
        rows = {}
        for key, per_iter_ns in _parse_nvtx_kern_sum(stats.stdout, args.iters).items():
            b = bytes_map.get(key)
            if b is None or per_iter_ns <= 0:
                continue
            rows[key] = (per_iter_ns / 1e3, b / per_iter_ns)  # us, GB/s
        if not rows:
            print(f"[warn] no QBENCH ranges parsed from nsys for backend={backend}.",
                  file=sys.stderr)
        elif out_dir:
            print(f"[nsys] saved {rep}.nsys-rep ({len(rows)} workloads)", file=sys.stderr)
        return rows
    finally:
        if tmp is not None:
            tmp.cleanup()          # temp path: report + SQLite removed
        else:
            # --nsys-out path: keep the raw .nsys-rep, but delete the generated
            # SQLite so a later run can never read stale data from it.
            sqlite = rep + ".sqlite"
            if os.path.exists(sqlite):
                os.remove(sqlite)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--backends", default="cpp,dsl",
                    help="Comma-separated: cpp (CUDA grouped kernel), dsl (CuTeDSL). "
                         "Default both.")
    ap.add_argument("--modes", default="gpu,cpu",
                    help="Comma-separated: gpu (kernel time), cpu (dispatch time). "
                         "Default both.")
    ap.add_argument("--groups", default=None,
                    help="Groups: preset names (--list-groups) and/or inline "
                         "'rep:MxN,MxN,...' with rep in sbd|vfd|vld. Separate groups "
                         "with ';' (',' also works between preset names).")
    ap.add_argument("--preset", default=None, choices=sorted(PRESETS),
                    help=f"Named group set: {sorted(PRESETS)}")
    ap.add_argument("--directions", default="row,col,both",
                    help="Comma-separated subset of row,col,both "
                         "(both = rowwise+colwise in one pass).")
    ap.add_argument("--in-dtypes", dest="in_dtypes", default="bf16",
                    help=f"Comma-separated input dtypes; 'all' = {_ALL_DTYPES}")
    ap.add_argument("--fp8s", default="e4m3",
                    help=f"Comma-separated FP8 output dtypes; 'all' = {_ALL_FP8}")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--all", action="store_true",
                    help="Every group x row/col/both x all input dtypes x both FP8 "
                         "formats. Very heavy — pair with modest --iters.")
    ap.add_argument("--list-groups", action="store_true",
                    help="Print the group presets and exit.")
    ap.add_argument("--nsys-out", dest="nsys_out", default=None,
                    help="Keep raw nsys artifacts in this dir (GPU mode): per backend "
                         "a .nsys-rep, its .nvtx_kern_sum.csv and a .log. "
                         "Default: discarded in a temp dir.")
    # Worker-only flags (not for interactive use).
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--backend", choices=["cpp", "dsl"], help=argparse.SUPPRESS)
    ap.add_argument("--gpu-nsys", dest="gpu_nsys", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--csv", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.list_groups:
        print("group presets (name: rep  members):")
        for name, (rep, shapes) in GROUP_PRESETS.items():
            shown = ", ".join(f"{m}x{n}" for m, n in shapes[:6])
            if len(shapes) > 6:
                shown += f", ... ({len(shapes)} members)"
            print(f"  {name:11s} {_REP_ALIAS[rep]:3s}  {shown}")
        print("\nnamed sets (--preset):")
        for name, groups in PRESETS.items():
            print(f"  {name:8s} {groups}")
        return 0

    # Resolve the axes shared by driver and worker.
    if args.all:
        args.groups = PRESETS["all"]
        args.directions = "row,col,both"
        args.in_dtypes, args.fp8s = ",".join(_ALL_DTYPES), ",".join(_ALL_FP8)
    else:
        if args.groups is None:
            args.groups = PRESETS[args.preset] if args.preset else _DEFAULT_GROUPS
        elif args.groups in PRESETS:
            args.groups = PRESETS[args.groups]
        if args.in_dtypes.strip() == "all":
            args.in_dtypes = ",".join(_ALL_DTYPES)
        if args.fp8s.strip() == "all":
            args.fp8s = ",".join(_ALL_FP8)

    # Validate the group specs up front, in the driver, so a typo fails fast
    # instead of inside a profiled subprocess.
    for spec in split_group_specs(args.groups):
        parse_group_spec(spec)
    for d in args.directions.split(","):
        assert d.strip() in ("row", "col", "both"), f"unknown direction {d!r}"

    if args.worker:
        return _worker_main(args)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for b in backends:
        assert b in ("cpp", "dsl"), f"unknown backend {b!r}"
    for m in modes:
        assert m in ("gpu", "cpu"), f"unknown mode {m!r}"

    if args.nsys_out:
        os.makedirs(args.nsys_out, exist_ok=True)
    nsys = shutil.which("nsys")
    if "gpu" in modes and nsys is None:
        print("[warn] nsys not found on PATH; skipping GPU mode (GPU timing comes "
              "from nsys). Install Nsight Systems or run --modes cpu.", file=sys.stderr)
        modes = [m for m in modes if m != "gpu"]

    # (backend, mode) -> {key: (us, gbps)}; key = (tag, M, N, dir).
    data, keys = {}, []
    for mode in modes:
        for backend in backends:
            rows = (_run_gpu_nsys_backend(nsys, backend, args, out_dir=args.nsys_out)
                    if mode == "gpu" else _run_cpu(backend, args))
            if rows:
                data[(backend, mode)] = rows
                for k in rows:
                    if k not in keys:
                        keys.append(k)

    if not data:
        print("No results (no backend ran successfully).", file=sys.stderr)
        return 1

    # Merged table: one row per (group tag, logical shape, dir); per mode show
    # cpp/dsl us and the cpp/dsl speedup (>1 == CuTeDSL faster). GB/s only for GPU
    # mode — it is meaningless for host dispatch time.
    print()
    header = f"{'tag':>40}  {'logical':>13}  {'dir':>4}"
    for mode in modes:
        m = mode.upper()
        header += f"  {m+'_cpp_us':>11}  {m+'_dsl_us':>11}  {m+'_x':>6}"
        if mode == "gpu":
            header += f"  {'cpp_GB/s':>9}  {'dsl_GB/s':>9}"
    print(header)
    print("-" * len(header))
    for tag, M, N, d in keys:
        line = f"{tag:>40}  {f'{M}x{N}':>13}  {d:>4}"
        for mode in modes:
            cpp_us, cpp_bw = data.get(("cpp", mode), {}).get((tag, M, N, d), (None, None))
            dsl_us, dsl_bw = data.get(("dsl", mode), {}).get((tag, M, N, d), (None, None))
            cpp_s = f"{cpp_us:11.2f}" if cpp_us is not None else f"{'-':>11}"
            dsl_s = f"{dsl_us:11.2f}" if dsl_us is not None else f"{'-':>11}"
            spd = f"{cpp_us / dsl_us:6.2f}" if (cpp_us and dsl_us) else f"{'-':>6}"
            line += f"  {cpp_s}  {dsl_s}  {spd}"
            if mode == "gpu":
                line += (f"  {cpp_bw:9.1f}" if cpp_bw is not None else f"  {'-':>9}")
                line += (f"  {dsl_bw:9.1f}" if dsl_bw is not None else f"  {'-':>9}")
        print(line)
    print("\n  us = microseconds/call; *_x = cpp/dsl speedup (>1 = CuTeDSL faster)")
    print("  GB/s = effective HBM bandwidth (in+out+scale bytes / GPU kernel time)")
    print("  GPU = kernel time from nsys nvtx_kern_sum (cold L2, all kernels in the")
    print("        range, incl. the CuTeDSL descriptor prologue); CPU = host dispatch")
    print("  tag = <group>_<rep><members>t_<in dtype>_<fp8>; logical = the 2D view")
    print("        the group is passed through (bookkeeping only for vld)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
