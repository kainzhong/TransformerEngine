#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""One-shot driver for the MXFP8 quantize benchmark.

`bench_mxfp8_cutedsl.py` measures ONE backend in ONE timing mode per process
(the backend is latched by NVTE_ENABLE_CUTEDSL_QUANT_BACKEND at import). This
driver runs every {backend} x {timing-mode} combination in its own subprocess
and prints a single merged table, so one command gives you the full picture:

    backend:  cpp = CUDA C++ kernels      (NVTE_ENABLE_CUTEDSL_QUANT_BACKEND=0)
              dsl = CuTeDSL kernels        (NVTE_ENABLE_CUTEDSL_QUANT_BACKEND=1)
    mode:     GPU = kernel time, cold L2   (CUDA events, --evict-l2)
              CPU = host dispatch time     (tight launch loop, --no-evict-l2)

Default (curated): combos {plain, dbias, gelu, dgelu} x directions {row, col}
(no bidimensional — no specialized kernel for it yet) x swizzle {off, on} x
bf16 x e4m3 x an LLM-representative shape set (hidden 4096-14336, a few thousand
tokens), for both backends and both modes. Override any axis (--preset/--shapes
for sizes), or use --all for the full matrix.

Usage:
    python run_mxfp8_benchmark.py                         # curated default
    python run_mxfp8_benchmark.py --preset llm --modes gpu
    python run_mxfp8_benchmark.py --combos plain --directions row --swizzle off
    python run_mxfp8_benchmark.py --backends dsl          # CuTeDSL only
    python run_mxfp8_benchmark.py --all --preset tiny     # everything
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH = Path(__file__).with_name("bench_mxfp8_cutedsl.py")
_MODE_FLAG = {"gpu": "--evict-l2", "cpu": "--no-evict-l2"}

# Full axes (mirror bench_mxfp8_cutedsl.py) — used to expand "all" / --all.
_ACTS = ["gelu", "silu", "relu", "qgelu", "srelu"]
_ALL_COMBOS = (["plain", "dbias"] + _ACTS
               + ["d" + a for a in _ACTS] + ["dbias_d" + a for a in _ACTS])
_ALL_DTYPES = ["bf16", "fp16", "fp32"]
_ALL_FP8 = ["e4m3", "e5m2"]

# Default shapes (tokens M x hidden N): LLM-representative — hidden dims 4096-
# 14336 (7B/70B hidden + Llama-3 MLP intermediate), a few thousand tokens. All
# multiples of 128 so the swizzled-scale layout applies. Override with --preset
# / --shapes.
_DEFAULT_SHAPES = "4096,4096;4096,8192;8192,8192;4096,14336"


def _expand(val, full):
    """Expand a comma list, turning the literal 'all' into the full axis."""
    if val is None:
        return None
    items = [v.strip() for v in val.split(",") if v.strip()]
    return ",".join(full) if items == ["all"] else val


def _detect_cute_dsl_arch():
    """sm_<major><minor>[a] for the current device (CuTeDSL compile target)."""
    try:
        import torch

        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}{'a' if major >= 9 else ''}"
    except Exception:
        return None


def _run_one(backend, mode, passthrough):
    """Run the bench for one (backend, mode); return {(tag, M, N, dir): us}."""
    env = dict(os.environ)
    env["NVTE_ENABLE_CUTEDSL_QUANT_BACKEND"] = "1" if backend == "dsl" else "0"
    if backend == "dsl" and "CUTE_DSL_ARCH" not in env:
        arch = _detect_cute_dsl_arch()
        if arch:
            env["CUTE_DSL_ARCH"] = arch

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    cmd = [sys.executable, str(BENCH), _MODE_FLAG[mode], "--csv", csv_path] + passthrough
    print(f"[run] backend={backend:3s} mode={mode}: {' '.join(cmd)}", file=sys.stderr)
    try:
        proc = subprocess.run(env=env, args=cmd, stdout=subprocess.DEVNULL)
        if proc.returncode != 0:
            print(f"[warn] backend={backend} mode={mode} exited {proc.returncode}; "
                  "skipping (is this backend available?)", file=sys.stderr)
            return None
        rows = {}
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                rows[(r["tag"], int(r["M"]), int(r["N"]), r["dir"])] = float(r["us"])
        return rows
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


def _fwd(args_ns, passthrough_keys):
    """Rebuild the forwarded bench CLI flags from parsed args."""
    out = []
    for flag, val in passthrough_keys.items():
        if val is None or val is False:
            continue
        if val is True:
            out.append(flag)
        else:
            out += [flag, str(val)]
    return out


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--backends", default="cpp,dsl",
                    help="Comma-separated: cpp (CUDA), dsl (CuTeDSL). Default both.")
    ap.add_argument("--modes", default="gpu,cpu",
                    help="Comma-separated: gpu (kernel time), cpu (dispatch time). Default both.")
    ap.add_argument("--all", action="store_true",
                    help="Override the curated defaults with EVERY case: all 17 "
                         "combos x row/col/both x all 3 input dtypes x both fp8 "
                         "formats x swizzle on+off. Very heavy — pair with a small "
                         "--preset and modest --iters.")
    # Curated defaults: plain + one act + one dact (+ plain dbias), rowwise and
    # columnwise (no bidimensional — no specialized kernel for it yet), swizzle
    # on+off, small preset. Override any axis explicitly; 'all' expands an axis.
    ap.add_argument("--combos", default="plain,dbias,gelu,dgelu")
    ap.add_argument("--directions", default="row,col",
                    help="Comma-separated subset of row,col,both.")
    ap.add_argument("--swizzle", choices=["off", "on", "both"], default="both",
                    help="Swizzled scale layout: off / on / both. Default both.")
    # Shapes: default to an LLM-representative set; --preset / --shapes override.
    ap.add_argument("--preset", default=None)
    # Forwarded to bench_mxfp8_cutedsl.py (see its --help for semantics).
    ap.add_argument("--shapes")
    ap.add_argument("--in-dtypes", dest="in_dtypes")
    ap.add_argument("--fp8s")
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--iters", type=int)
    args = ap.parse_args()

    # Default to the LLM-representative shape set unless the user picked shapes.
    if args.preset is None and args.shapes is None:
        args.shapes = _DEFAULT_SHAPES

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for b in backends:
        assert b in ("cpp", "dsl"), f"unknown backend {b!r}"
    for m in modes:
        assert m in _MODE_FLAG, f"unknown mode {m!r}"

    # --all expands every axis; otherwise honor what's given (expanding any
    # explicit "all" literal per axis).
    if args.all:
        combos, in_dtypes, fp8s = (",".join(_ALL_COMBOS),
                                   ",".join(_ALL_DTYPES), ",".join(_ALL_FP8))
        directions, swizzles = ["row", "col", "both"], [False, True]
    else:
        combos = _expand(args.combos, _ALL_COMBOS)
        in_dtypes = _expand(args.in_dtypes, _ALL_DTYPES)
        fp8s = _expand(args.fp8s, _ALL_FP8)
        directions = [d.strip() for d in args.directions.split(",") if d.strip()]
        swizzles = {"off": [False], "on": [True], "both": [False, True]}[args.swizzle]
    for d in directions:
        assert d in ("row", "col", "both"), f"unknown direction {d!r}"

    base = _fwd(args, {
        "--preset": args.preset, "--shapes": args.shapes, "--combos": combos,
        "--in-dtypes": in_dtypes, "--fp8s": fp8s,
        "--warmup": args.warmup, "--iters": args.iters,
    })

    # (backend, mode) -> {key: us}  (key/tag already encodes combo/dtype/fp8/swizzle;
    # direction is part of the row key). Sweep direction + swizzle here since the
    # bench takes one direction and one swizzle setting per process.
    data = {}
    keys = []
    for mode in modes:
        for backend in backends:
            for direction in directions:
                for swizzle in swizzles:
                    passthrough = (base + ["--direction", direction]
                                   + (["--swizzle"] if swizzle else []))
                    rows = _run_one(backend, mode, passthrough)
                    if rows is None:
                        continue
                    data.setdefault((backend, mode), {}).update(rows)
                    for k in rows:
                        if k not in keys:
                            keys.append(k)

    if not data:
        print("No results (no backend ran successfully).", file=sys.stderr)
        return 1

    # Merged table: one row per (tag, shape, dir); per mode show cpp/dsl us and
    # the cpp/dsl speedup (>1 == CuTeDSL faster).
    print()
    header = f"{'tag':>28}  {'shape':>11}  {'dir':>4}"
    for mode in modes:
        m = mode.upper()
        header += f"  {m+'_cpp_us':>11}  {m+'_dsl_us':>11}  {m+'_x':>6}"
    print(header)
    print("-" * len(header))
    for tag, M, N, d in keys:
        line = f"{tag:>28}  {f'{M}x{N}':>11}  {d:>4}"
        for mode in modes:
            cpp = data.get(("cpp", mode), {}).get((tag, M, N, d))
            dsl = data.get(("dsl", mode), {}).get((tag, M, N, d))
            cpp_s = f"{cpp:11.2f}" if cpp is not None else f"{'-':>11}"
            dsl_s = f"{dsl:11.2f}" if dsl is not None else f"{'-':>11}"
            spd = f"{cpp / dsl:6.2f}" if (cpp and dsl) else f"{'-':>6}"
            line += f"  {cpp_s}  {dsl_s}  {spd}"
        print(line)
    print("\n  us = microseconds/call; *_x = cpp/dsl speedup (>1 = CuTeDSL faster)")
    print("  GPU = kernel time (cold L2, CUDA events); CPU = host dispatch time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
