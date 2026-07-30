# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Drive bench_mxfp8_quantize.py under nsys and emit a kernel perf summary.

    python profile_mxfp8_quantize.py --label whatever

One run, one build, one set of numbers -- this script never compares runs. Each
run's artifacts carry the branch, commit and build timestamp of the extension
that produced them (printed at the top, and stamped in a `build`/`built` column
on every CSV row), so re-running on the same checkout is self-evident rather
than something you have to remember.

The workload script wraps each op call in a same-named NVTX range, so the real
per-workload GPU kernel time comes from nsys's `nvtx_kern_sum` report; the
aggregate per-kernel view comes from `cuda_gpu_kern_sum`. Nothing is timed on
the host, so there is no CUDA-event or wall-clock skew in the numbers.

Outputs, all under --outdir (default: ./mxfp8_perf):
    <label>.nsys-rep         raw nsys capture
    <label>_manifest.json    workload -> byte-count map + build provenance
    <label>_workloads.csv    per-workload kernel time + achieved GB/s
    <label>_kernels.csv      per-kernel-name aggregate
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench_mxfp8_quantize.py")


def sh(cmd, **kw):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False, **kw)


def _find_col(header, *candidates):
    """Return index of the first header cell containing any candidate substring."""
    low = [h.lower() for h in header]
    for cand in candidates:
        for i, h in enumerate(low):
            if cand in h:
                return i
    return None


def _read_stats_csv(path):
    """nsys stats CSVs sometimes carry preamble lines before the real header."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    for i, row in enumerate(rows):
        if len(row) > 2 and any("time" in c.lower() for c in row):
            return row, [r for r in rows[i + 1 :] if len(r) == len(row)]
    return None, []


def _to_float(s):
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return None


def run_nsys(label, outdir, bench_args, env_extra=None):
    os.makedirs(outdir, exist_ok=True)
    rep = os.path.join(outdir, f"{label}.nsys-rep")
    manifest = os.path.join(outdir, f"{label}_manifest.json")
    for stale in (rep, rep.replace(".nsys-rep", ".sqlite")):
        if os.path.exists(stale):
            os.remove(stale)

    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # `python <script>` puts the SCRIPT's directory on sys.path -- NOT the cwd -- so
    # the local build no longer shadows the prebuilt transformer_engine in
    # dist-packages. Without this the profile silently measures the container's TE.
    repo = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    if env_extra:
        env.update(env_extra)

    cmd = [
        "nsys", "profile",
        "-t", "cuda,nvtx",
        "-s", "none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        "--resolve-symbols=false",  # nsys finalize hangs on this sbsa box otherwise
        "-o", rep.replace(".nsys-rep", ""),
        sys.executable, BENCH,
        "--manifest", manifest,
    ] + bench_args
    r = sh(cmd, env=env)
    if r.returncode != 0 or not os.path.exists(rep):
        print("nsys profile failed", file=sys.stderr)
        return None, None
    return rep, manifest


def nsys_stats(rep, report, outdir, label):
    base = os.path.join(outdir, f"{label}_{report}")
    r = sh([
        "nsys", "stats", "--force-export=true", "-r", report,
        "-f", "csv", "-o", base, rep,
    ], capture_output=True, text=True)
    # nsys appends "_<report>.csv" to -o
    for cand in (f"{base}_{report}.csv", f"{base}.csv"):
        if os.path.exists(cand):
            return cand
    print(r.stdout or "", r.stderr or "", file=sys.stderr)
    return None


def summarize(label, outdir, rep, manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)
    by_range = {w["range"]: w for w in manifest["workloads"]}
    prov = manifest.get("provenance", {})
    # Stamped into every row so a CSV can never be ambiguous about its build.
    build_id = f'{prov.get("git_branch", "?")}@{prov.get("git_commit", "?")}' + (
        "-dirty" if prov.get("git_dirty") else "")

    # ---- per-workload (NVTX range) kernel time --------------------------------
    # nvtx_kern_sum emits one row per (range, kernel). A fused op can launch more
    # than one kernel inside the range (e.g. dbias adds reduce_dbias_kernel), so
    # sum every kernel's total time in the range and divide by the NVTX instance
    # count to get GPU time per iteration for the whole op.
    agg = {}  # range -> {"total_ns", "nvtx_inst", "kernels": {name: ns}}
    nvtx_csv = nsys_stats(rep, "nvtx_kern_sum", outdir, label)
    if nvtx_csv:
        header, rows = _read_stats_csv(nvtx_csv)
        if header:
            i_range = _find_col(header, "range")
            i_tot = _find_col(header, "total time")
            i_nvtx = _find_col(header, "nvtx inst")
            i_kern = _find_col(header, "kernel name", "operation", "name")
            for row in rows:
                # The range column is a colon-separated NVTX stack path; ours are
                # leaves under a blank root, so the name is the last segment.
                rng = row[i_range].strip().split(":")[-1].strip()
                if rng not in by_range:
                    continue
                tot = _to_float(row[i_tot]) or 0.0
                ninst = _to_float(row[i_nvtx]) if i_nvtx is not None else None
                e = agg.setdefault(rng, {"total_ns": 0.0, "nvtx_inst": 0.0, "kernels": {}})
                e["total_ns"] += tot
                e["nvtx_inst"] = max(e["nvtx_inst"], ninst or 0.0)
                kname = row[i_kern].strip() if i_kern is not None else ""
                e["kernels"][kname] = e["kernels"].get(kname, 0.0) + tot

    wl_rows = []
    for rng, e in agg.items():
        wl = by_range[rng]
        inst = e["nvtx_inst"] or wl["iters"]
        us = e["total_ns"] / inst / 1e3
        if us <= 0:
            continue
        main_kernel = max(e["kernels"].items(), key=lambda kv: kv[1])[0] if e["kernels"] else ""
        wl_rows.append({
            "build": build_id,
            "built": prov.get("te_core_built", prov.get("te_ext_built", "?")),
            "combo": wl["combo"], "dir": wl["dir"],
            "M": wl["M"], "N": wl["N"],
            "shape": f'{wl["M"]}x{wl["N"]}',
            "swizzle": wl["swizzle"],
            "us": us,
            "gbps": wl["bytes"] / (us * 1e-6) / 1e9,
            "bytes": wl["bytes"],
            "n_kernels": len(e["kernels"]),
            "kernel": main_kernel,
        })

    wl_csv = os.path.join(outdir, f"{label}_workloads.csv")
    if wl_rows:
        # deterministic order matching the bench sweep
        order = {w["range"]: i for i, w in enumerate(manifest["workloads"])}
        wl_rows.sort(key=lambda r: order.get(
            f'Q|{r["combo"]}|{r["dir"]}|{r["shape"]}' + ("|sw" if r["swizzle"] else ""), 1 << 30))
        with open(wl_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(wl_rows[0].keys()))
            w.writeheader()
            w.writerows(wl_rows)

    # ---- per-kernel-name aggregate -------------------------------------------
    k_rows = []
    kern_csv_in = nsys_stats(rep, "cuda_gpu_kern_sum", outdir, label)
    if kern_csv_in:
        header, rows = _read_stats_csv(kern_csv_in)
        if header:
            i_name = _find_col(header, "name")
            i_inst = _find_col(header, "instances", "count")
            i_avg = _find_col(header, "avg")
            i_tot = _find_col(header, "total time")
            i_pct = _find_col(header, "time (%)", "%")
            for row in rows:
                name = row[i_name].strip()
                k_rows.append({
                    "build": build_id,
                    "kernel": name,
                    "instances": _to_float(row[i_inst]) if i_inst is not None else None,
                    "avg_us": (_to_float(row[i_avg]) or 0) / 1e3 if i_avg is not None else None,
                    "total_ms": (_to_float(row[i_tot]) or 0) / 1e6 if i_tot is not None else None,
                    "pct": _to_float(row[i_pct]) if i_pct is not None else None,
                })
    k_csv = os.path.join(outdir, f"{label}_kernels.csv")
    if k_rows:
        with open(k_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(k_rows[0].keys()))
            w.writeheader()
            w.writerows(k_rows)

    # ---- print ---------------------------------------------------------------
    print(f"\n=== {label}: build under test ===")
    print(f"  git       {build_id}")
    print(f"  kernels   {prov.get('te_core', '?')}")
    print(f"  built     {prov.get('te_core_built', '?')}   <- the one that matters")
    print(f"  pybind    {prov.get('te_ext', '?')}")
    print(f"  built     {prov.get('te_ext_built', '?')}")
    print(f"  gpu       {prov.get('gpu', '?')}   torch {prov.get('torch', '?')}")

    print(f"\n=== {label}: per-workload GPU kernel time (cold L2, from nsys NVTX) ===")
    print(f"{'combo':>14} {'dir':>5} {'shape':>13} {'us':>10} {'GB/s':>9}   kernel")
    print("-" * 100)
    for r in wl_rows:
        print(f'{r["combo"]:>14} {r["dir"]:>5} {r["shape"]:>13} '
              f'{r["us"]:10.2f} {r["gbps"]:9.1f}   {_short_kernel(r["kernel"])}')

    print(f"\n=== {label}: aggregate per kernel ===")
    print(f"{'instances':>10} {'avg_us':>10} {'total_ms':>10}   kernel")
    print("-" * 100)
    for r in sorted(k_rows, key=lambda r: -(r["total_ms"] or 0))[:15]:
        print(f'{int(r["instances"] or 0):10d} {r["avg_us"] or 0:10.2f} '
              f'{r["total_ms"] or 0:10.2f}   {_short_kernel(r["kernel"])}')

    print(f"\nwritten: {wl_csv}\n         {k_csv}")
    return wl_csv


_NS_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*::")


def _short_kernel(name, width=96):
    """Strip `void `, namespaces and the runtime-arg list; keep template args,
    which are what distinguish the MXFP8 kernel instantiations from each other."""
    if not name:
        return ""
    s = name[5:] if name.startswith("void ") else name
    # Drop the trailing "(...)" runtime parameter list, keeping template args.
    depth = 0
    for i, ch in enumerate(s):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            s = s[:i]
            break
    s = _NS_RE.sub("", s)
    s = re.sub(r"\(bool\)", "", s)
    s = re.sub(r"\(unsigned long\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= width else s[: width - 1] + "…"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--label", default="run", help="Name for this run's artifacts")
    p.add_argument("--outdir", default=os.path.join(HERE, "mxfp8_perf"))
    p.add_argument("--summarize-only", metavar="NSYS_REP",
                   help="Skip profiling; re-summarize an existing .nsys-rep")
    args, bench_args = p.parse_known_args()

    if not shutil.which("nsys"):
        print("nsys not found on PATH", file=sys.stderr)
        return 1

    if args.summarize_only:
        manifest = os.path.join(args.outdir, f"{args.label}_manifest.json")
        return 0 if summarize(args.label, args.outdir, args.summarize_only, manifest) else 1

    rep, manifest = run_nsys(args.label, args.outdir, bench_args)
    if rep is None:
        return 1
    summarize(args.label, args.outdir, rep, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
