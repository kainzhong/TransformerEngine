# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark the NVFP4 quantize-transpose kernels.

Same shape as bench_mxfp8_quantize.py -- it does no timing of its own, it just
wraps each op call in a same-named NVTX range so nsys can attribute real CUPTI
kernel durations per workload. Drive it with profile_mxfp8_quantize.py
(--bench selects this file) and read the results with proof/nsys_load.py.

Covers both dispatch paths in cast/nvfp4/quantize_transpose_nvfp4.cuh:
  1D  quantize_transpose_nvfp4_kernel      (default)
  2D  quantize_transpose_nvfp4_2D_kernel   (with_2d_quantization=True)
"""

import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys

import torch
import torch.cuda.nvtx as nvtx

import transformer_engine.pytorch as te  # must precede transformer_engine_torch
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer

HERE = os.path.dirname(os.path.abspath(__file__))


def mapped_lib():
    for line in open("/proc/self/maps"):
        if "libtransformer_engine" in line:
            return line.split()[-1]
    return "?"


def mapped_lib_md5():
    p = mapped_lib()
    if p == "?" or not os.path.exists(p):
        return "?"
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_provenance():
    repo = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

    def git(*a):
        try:
            return subprocess.run(["git", "-C", repo, *a], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:  # pylint: disable=broad-except
            return "?"

    return {
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "mapped_lib": mapped_lib(),
        "mapped_lib_md5": mapped_lib_md5(),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }


SHAPES = [
    (4096, 4096),
    (4096, 14336),
    (8192, 8192),
    (8192, 28672),
    (2048, 12288),
    (16384, 5120),
]

_DIR = {"row": (True, False), "col": (False, True), "both": (True, True)}

_L2 = None


def l2_evict():
    global _L2
    if _L2 is None:
        _L2 = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    return _L2


def workload_bytes(M, N, esz, rowwise, colwise):
    """inputs read + FP4 data (2 elems/byte) + e4m3 block scales written."""
    total = M * N * esz
    if rowwise:
        total += M * N // 2 + M * (N // 16)
    if colwise:
        total += M * N // 2 + (M // 16) * N
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default=None, help="'M,N;M,N;...' (default: the llm preset)")
    p.add_argument("--directions", default="row,col,both")
    p.add_argument("--modes", default="1d,2d", help="1d and/or 2d quantization")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--no-evict", action="store_true", help="keep L2 warm between iterations")
    p.add_argument("--manifest", default=None)
    args = p.parse_args()

    shapes = (SHAPES if not args.shapes else
              [tuple(int(v) for v in t.split(",")) for t in args.shapes.split(";") if t.strip()])
    dirs = [d.strip() for d in args.directions.split(",") if d.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    prov = build_provenance()
    print("build under test:")
    for k, v in prov.items():
        print(f"  {k:16} {v}")
    print()

    manifest = []
    evict = None if args.no_evict else l2_evict()

    for mode in modes:
        for M, N in shapes:
            for d in dirs:
                rowwise, colwise = _DIR[d]
                torch.manual_seed(0)
                x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
                q = NVFP4Quantizer(
                    fp4_dtype=tex.DType.kFloat4E2M1,
                    rowwise=rowwise,
                    columnwise=colwise,
                    with_2d_quantization=(mode == "2d"),
                )
                q.internal = True
                tag = f"nvfp4{mode}|{d}|{M}x{N}"
                rng = "Q|" + tag
                try:
                    tex.quantize(x, q)
                    torch.cuda.synchronize()
                except Exception as e:  # pylint: disable=broad-except
                    print(f"SKIP {tag}: {type(e).__name__}: {str(e).splitlines()[0]}", flush=True)
                    del x
                    torch.cuda.empty_cache()
                    continue

                nbytes = workload_bytes(M, N, x.element_size(), rowwise, colwise)
                manifest.append({"range": rng, "combo": f"nvfp4{mode}", "dir": d, "M": M, "N": N,
                                 "swizzle": False, "bytes": nbytes, "iters": args.iters})
                print(f"WORKLOAD {rng} bytes={nbytes} iters={args.iters}", flush=True)

                for _ in range(args.warmup):
                    tex.quantize(x, q)
                torch.cuda.synchronize()
                for _ in range(args.iters):
                    if evict is not None:
                        evict.zero_()
                    torch.cuda.synchronize()
                    nvtx.range_push(rng)
                    tex.quantize(x, q)
                    torch.cuda.synchronize()
                    nvtx.range_pop()

                del x
                torch.cuda.empty_cache()

    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump({"provenance": prov, "workloads": manifest}, f, indent=2)
        print(f"\nmanifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
