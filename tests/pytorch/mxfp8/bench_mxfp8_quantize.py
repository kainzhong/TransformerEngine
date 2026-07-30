# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark the MXFP8 quantize kernels over representative LLM shapes.

This is the *workload* half of the harness. It does no timing of its own: it
warms up, then runs `iters` cold-L2 iterations of each workload with the single
op call wrapped in a same-named NVTX range, so that nsys can attribute real
CUPTI kernel durations per workload (the kernel name encodes the template
config but NOT the shape, so ranges are what separate the workloads).

Run it through `profile_mxfp8_quantize.py`, which drives nsys and parses the
`nvtx_kern_sum` / `cuda_gpu_kern_sum` reports.

Covered fusions (all of which route through the MXFP8 quantize dispatch in
transformer_engine/common/cast/mxfp8/):

  plain       tex.quantize(x)                 cast only
  <act>       tex.<act>(x, q)                 ACT + quantize          (fwd MLP)
  d<act>      tex.d<act>(grad, x, q)          DACT + quantize         (bwd MLP)
  dbias       tex.bgrad_quantize(grad, q)     DBIAS + quantize        (bwd bias)
  dbias_d<act> tex.dbias_d<act>(grad, x, q)   DBIAS + DACT + quantize (bwd MLP)

NOTE on which kernel each combo lands on: `plain` without swizzled scales hits
the *specialized* cast-only kernel (specialized::hasSpec is true only when
IS_DBIAS=IS_DACT=IS_ACT=false) for the rowwise and bidimensional cases. Every
ACT/DACT/DBIAS combo -- and plain with GEMM-swizzled scales -- forces the
*generic* `quantize_mxfp8_kernel`. The generic kernel is the one touched by the
shared-memory-alignment change under test, so those combos are the signal;
plain row/both is a control that should not move.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

import torch
import torch.cuda.nvtx as nvtx

import transformer_engine.pytorch as te  # must precede transformer_engine_torch
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

HERE = os.path.dirname(os.path.abspath(__file__))


def mapped_lib():
    """Path of the libtransformer_engine.so actually mapped into THIS process."""
    for line in open("/proc/self/maps"):
        if "libtransformer_engine" in line:
            return line.split()[-1]
    return "?"


def mapped_lib_md5():
    """Content hash of that library -- the only thing that truly names the build.

    mtime is not enough: the pybind extension relinks on every build while the
    CUDA library may not, which is exactly how an A/B can end up comparing one
    build against itself.
    """
    import hashlib

    p = mapped_lib()
    if p == "?" or not os.path.exists(p):
        return "?"
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_provenance():
    """Identify the build that is actually loaded in THIS process.

    Two runs of the same sweep are indistinguishable from their numbers alone,
    so every artifact records which checkout and which compiled extension
    produced it. The extension's mtime is the load-bearing field: a stale build
    (checkout switched but not rebuilt) shows an mtime older than the checkout,
    which is the failure mode that silently corrupts an A/B.
    """
    repo = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

    def git(*a):
        try:
            return subprocess.run(["git", "-C", repo, *a], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:  # pylint: disable=broad-except
            return "?"

    def mtime(path):
        if path and os.path.exists(path):
            return datetime.datetime.fromtimestamp(
                os.path.getmtime(path)).isoformat(timespec="seconds")
        return "?"

    # The pybind extension is NOT where the CUDA kernels live -- libtransformer_engine.so
    # is. Timestamping only the extension would happily call a build "fresh" while the
    # kernels were stale, so record both and let the reader see them disagree.
    ext = getattr(tex, "__file__", None) or "?"
    core = os.path.join(repo, "libtransformer_engine.so")

    return {
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "te_ext": ext,
        "te_ext_built": mtime(ext),
        "te_core": core if os.path.exists(core) else "?",
        "te_core_built": mtime(core),
        "mapped_lib": mapped_lib(),
        "mapped_lib_md5": mapped_lib_md5(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }


def print_provenance(prov):
    print("build under test:")
    print(f"  git       {prov['git_branch']} @ {prov['git_commit']}"
          f"{'  (DIRTY)' if prov['git_dirty'] else ''}")
    print(f"  kernels   {prov['te_core']}")
    print(f"  built     {prov['te_core_built']}   <- the one that matters")
    print(f"  pybind    {prov['te_ext']}")
    print(f"  built     {prov['te_ext_built']}")
    print(f"  mapped    {prov['mapped_lib']}")
    print(f"  md5       {prov['mapped_lib_md5']}   <- names the build")
    print(f"  gpu       {prov['gpu']}   torch {prov['torch']}")


# (M, N) = (tokens, hidden). Tokens = batch*seq for a typical training micro-batch.
# Each pair is a tensor that really gets MXFP8-quantized in a transformer step.
SHAPE_PRESETS = {
    # Hidden-dim and FFN-intermediate tensors from real model configs.
    "llm": [
        (4096, 4096),  # Llama-3-8B  hidden      (4k tokens)
        (4096, 14336),  # Llama-3-8B  FFN inter.
        (8192, 8192),  # Llama-3-70B hidden      (8k tokens)
        (8192, 28672),  # Llama-3-70B FFN inter.
        (2048, 12288),  # GPT-3-175B  hidden
        (16384, 5120),  # long-seq / Nemotron-ish hidden
    ],
    # Small sweep for a quick sanity run.
    "quick": [(4096, 4096), (8192, 28672)],
    # Aspect-ratio stress: tall-narrow vs short-wide.
    "aspect": [(1024, 16384), (16384, 1024), (65536, 1024), (1024, 65536)],
}

_ACT_FNS = {a: getattr(tex, a) for a in ("gelu", "silu", "relu", "qgelu", "srelu")}
_DACT_FNS = {"d" + a: getattr(tex, "d" + a) for a in _ACT_FNS}
_DBIAS_DACT_FNS = {"dbias_d" + a: getattr(tex, "dbias_d" + a) for a in _ACT_FNS}

# combo -> kind. `kind` picks the tex entry point and how many inputs are read.
COMBO_KIND = {"plain": "plain", "dbias": "dbias"}
for _a in _ACT_FNS:
    COMBO_KIND[_a] = "act"
    COMBO_KIND["d" + _a] = "dact"
    COMBO_KIND["dbias_d" + _a] = "dbias_dact"

# Default combo set: one representative of each fusion shape, plus the two
# activations that actually show up in modern LLMs (gelu, silu/swiglu).
DEFAULT_COMBOS = ["plain", "gelu", "silu", "dgelu", "dsilu", "dbias", "dbias_dgelu"]

_FP8_DTYPES = {"e4m3": tex.DType.kFloat8E4M3, "e5m2": tex.DType.kFloat8E5M2}
_TORCH_IN_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

_DIR_RCW = {"row": (True, False), "col": (False, True), "both": (True, True)}


def _make_quantizer(rowwise, colwise, fp8_dtype, swizzle):
    q = MXFP8Quantizer(
        fp8_dtype=_FP8_DTYPES[fp8_dtype],
        rowwise=rowwise,
        columnwise=colwise,
    )
    q.internal = True  # skip the Float8Tensor wrapper / python overhead
    if swizzle:
        q.optimize_for_gemm = True
    return q


def make_fn(combo, x, act_input, quantizer):
    """Return a 0-arg callable performing one fused quantize."""
    kind = COMBO_KIND[combo]
    if kind == "plain":
        return lambda: tex.quantize(x, quantizer)
    if kind == "act":
        op = _ACT_FNS[combo]
        return lambda: op(x, quantizer)
    if kind == "dact":
        op = _DACT_FNS[combo]
        return lambda: op(x, act_input, quantizer)  # x is grad
    if kind == "dbias":
        return lambda: tex.bgrad_quantize(x, quantizer)  # x is grad
    if kind == "dbias_dact":
        op = _DBIAS_DACT_FNS[combo]
        return lambda: op(x, act_input, quantizer)  # x is grad
    raise ValueError(f"unknown combo {combo!r}")


def workload_bytes(M, N, in_bytes_per_elt, need_act_input, rowwise, colwise):
    """Lower bound on HBM traffic: inputs read + FP8 out + e8m0 scales written."""
    total = M * N * in_bytes_per_elt * (2 if need_act_input else 1)
    if rowwise:
        total += M * N  # rowwise fp8 data
        total += M * (N // 32)  # rowwise e8m0 scales
    if colwise:
        total += M * N  # colwise fp8 data
        total += (M // 32) * N  # colwise e8m0 scales
    return total


# 512 MB f32 evict buffer -- comfortably larger than Blackwell's L2.
_L2_EVICT_BUF = None


def _l2_evict_buf():
    global _L2_EVICT_BUF
    if _L2_EVICT_BUF is None:
        _L2_EVICT_BUF = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    return _L2_EVICT_BUF


def run_workload(M, N, dir_name, combo, args, manifest):
    rowwise, colwise = _DIR_RCW[dir_name]
    in_dt = _TORCH_IN_DTYPES[args.in_dtype]
    need_act_input = COMBO_KIND[combo] in ("dact", "dbias_dact")

    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=in_dt, device="cuda")
    act_input = torch.randn(M, N, dtype=in_dt, device="cuda") if need_act_input else None
    quantizer = _make_quantizer(rowwise, colwise, args.fp8, args.swizzle)
    fn = make_fn(combo, x, act_input, quantizer)

    tag = f"{combo}|{dir_name}|{M}x{N}"
    if args.swizzle:
        tag += "|sw"
    rng = "Q|" + tag

    # Probe once outside any range -- also surfaces unsupported configs early.
    try:
        fn()
        torch.cuda.synchronize()
    except Exception as e:  # pylint: disable=broad-except
        print(f"SKIP {tag}: {type(e).__name__}: {str(e).splitlines()[0]}", flush=True)
        del x, act_input
        torch.cuda.empty_cache()
        return

    nbytes = workload_bytes(M, N, x.element_size(), need_act_input, rowwise, colwise)
    manifest.append(
        {
            "range": rng,
            "combo": combo,
            "dir": dir_name,
            "M": M,
            "N": N,
            "in_dtype": args.in_dtype,
            "fp8": args.fp8,
            "swizzle": bool(args.swizzle),
            "bytes": nbytes,
            "iters": args.iters,
        }
    )
    print(f"WORKLOAD {rng} bytes={nbytes} iters={args.iters}", flush=True)

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()

    evict = None if args.no_evict else _l2_evict_buf()
    for _ in range(args.iters):
        if evict is not None:
            evict.zero_()  # flush L2 -- OUTSIDE the measured range
        torch.cuda.synchronize()
        nvtx.range_push(rng)  # same name every iter -> nsys aggregates
        fn()
        torch.cuda.synchronize()
        nvtx.range_pop()

    del x, act_input
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preset", default="llm", choices=sorted(SHAPE_PRESETS))
    p.add_argument("--shapes", default=None, help="Custom shapes 'M,N;M,N;...' (overrides preset)")
    p.add_argument("--combos", default=None, help=f"Comma-separated; default {DEFAULT_COMBOS}")
    p.add_argument("--directions", default="row,col,both")
    p.add_argument("--in-dtype", default="bf16", choices=sorted(_TORCH_IN_DTYPES))
    p.add_argument("--fp8", default="e4m3", choices=sorted(_FP8_DTYPES))
    p.add_argument("--swizzle", action="store_true", help="GEMM-swizzled scales")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--no-evict", action="store_true",
                   help="do NOT flush L2 between iterations (warm-cache steady state)")
    p.add_argument("--manifest", default=None, help="Write workload manifest JSON here")
    args = p.parse_args()

    if args.shapes:
        shapes = [tuple(int(v) for v in s.split(",")) for s in args.shapes.split(";") if s.strip()]
    else:
        shapes = SHAPE_PRESETS[args.preset]

    combos = [c.strip() for c in args.combos.split(",")] if args.combos else list(DEFAULT_COMBOS)
    for c in combos:
        if c not in COMBO_KIND:
            print(f"unknown combo: {c}", file=sys.stderr)
            return 1
    dirs = [d.strip() for d in args.directions.split(",") if d.strip()]
    for d in dirs:
        if d not in _DIR_RCW:
            print(f"unknown direction: {d}", file=sys.stderr)
            return 1

    prov = build_provenance()
    print_provenance(prov)
    print(f"shapes={shapes}\ncombos={combos}\ndirs={dirs}  in={args.in_dtype} fp8={args.fp8} "
          f"swizzle={args.swizzle}  warmup={args.warmup} iters={args.iters}\n", flush=True)

    manifest = []
    for combo in combos:
        for M, N in shapes:
            for d in dirs:
                run_workload(M, N, d, combo, args, manifest)

    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump({"provenance": prov, "workloads": manifest}, f, indent=2)
        print(f"\nmanifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
