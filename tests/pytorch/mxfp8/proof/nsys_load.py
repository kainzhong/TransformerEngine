# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Load pure-GPU kernel times out of an nsys `nvtx_kern_sum` report.

Only the MXFP8 quantize kernel is counted. `bgrad_quantize` also launches a
separate `reduce_dbias_kernel`, which this change does not touch -- including it
would just dilute the ratio with time that cannot possibly have moved.

Everything here is CUPTI kernel duration. No host time, no CUDA-event skew.
"""

import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NSYS = os.path.join(HERE, "nsys")

KERNEL_MATCH = "mxfp8"  # matches both the generic and the specialized cast-only kernel


def _col(header, *cands):
    low = [h.lower() for h in header]
    for c in cands:
        for i, h in enumerate(low):
            if c in h:
                return i
    raise KeyError(cands)


def load(label, nsys_dir=NSYS):
    """label -> {key: dict(us, gbps, bytes, combo, dir, M, N, kernel, specialized)}"""
    hits = glob.glob(os.path.join(nsys_dir, f"{label}_nvtx_kern_sum*.csv"))
    if not hits:
        raise FileNotFoundError(f"no nvtx_kern_sum csv for {label} in {nsys_dir}")
    rows = list(csv.reader(open(hits[0], newline="")))
    hdr = next(r for r in rows if any("kernel name" in c.lower() for c in r))
    i_rng = _col(hdr, "nvtx range")
    i_med = _col(hdr, "med (ns)", "median")
    i_knm = _col(hdr, "kernel name")

    man = json.load(open(os.path.join(nsys_dir, f"{label}_manifest.json")))
    meta = {w["range"]: w for w in man["workloads"]}

    out = {}
    for r in rows[rows.index(hdr) + 1 :]:
        if len(r) <= max(i_rng, i_med, i_knm):
            continue
        # nsys renders the range path with ':' separators, e.g. ":Q|plain|col|4096x4096"
        rng, kern = r[i_rng].strip().lstrip(":"), r[i_knm]
        if not rng or KERNEL_MATCH not in kern:
            continue
        w = meta.get(rng)
        if w is None:
            continue
        us = float(r[i_med]) / 1000.0
        key = rng[2:] if rng.startswith("Q|") else rng  # strip the "Q|" prefix
        prev = out.get(key)
        if prev:  # more than one matching kernel in the range: sum them
            us += prev["us"]
        out[key] = {
            "us": us,
            "bytes": w["bytes"],
            "gbps": w["bytes"] / (us * 1e-6) / 1e9,
            "combo": w["combo"],
            "dir": w["dir"],
            "M": w["M"],
            "N": w["N"],
            "kernel": kern,
            "specialized": "cast_only" in kern,
        }
    return out, man["provenance"]
