# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark the FP8 gated cast kernel (cast/fp8/gated_fp8.cuh).

Same NVTX/manifest contract as bench_mxfp8_quantize.py, so proof/nsys_load.py
reads it unchanged. FP8 here is per-tensor scaling, so there is no
rowwise/columnwise axis -- only the fused activation and the shape.
"""

import argparse
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
from transformer_engine.pytorch import Float8Quantizer

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

    return {"git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_commit": git("rev-parse", "--short", "HEAD"),
            "mapped_lib": mapped_lib(), "mapped_lib_md5": mapped_lib_md5(),
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}


SHAPES = [(4096, 4096), (4096, 14336), (8192, 8192), (8192, 28672), (2048, 12288), (16384, 5120)]

_L2 = None


def l2_evict():
    global _L2
    if _L2 is None:
        _L2 = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    return _L2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default=None)
    p.add_argument("--combos", default="swiglu,geglu,dswiglu,dgeglu")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--no-evict", action="store_true")
    p.add_argument("--manifest", default=None)
    args = p.parse_args()

    shapes = (SHAPES if not args.shapes else
              [tuple(int(v) for v in t.split(",")) for t in args.shapes.split(";") if t.strip()])
    combos = [c.strip() for c in args.combos.split(",") if c.strip()]

    prov = build_provenance()
    print("build under test:")
    for k, v in prov.items():
        print(f"  {k:16} {v}")
    print()

    manifest = []
    evict = None if args.no_evict else l2_evict()

    for combo in combos:
        bwd = combo.startswith("d")
        op = getattr(tex, combo)
        for M, N in shapes:
            torch.manual_seed(0)
            x = torch.randn(M, 2 * N, dtype=torch.bfloat16, device="cuda")
            g = torch.randn(M, N, dtype=torch.bfloat16, device="cuda") if bwd else None
            q = Float8Quantizer(scale=torch.ones(1, device="cuda"),
                                amax=torch.zeros(1, device="cuda"),
                                fp8_dtype=tex.DType.kFloat8E4M3)
            fn = (lambda: op(g, x, q)) if bwd else (lambda: op(x, q))
            tag = f"{combo}|na|{M}x{N}"
            rng = "Q|" + tag
            try:
                fn()
                torch.cuda.synchronize()
            except Exception as e:  # pylint: disable=broad-except
                print(f"SKIP {tag}: {type(e).__name__}: {str(e).splitlines()[0]}", flush=True)
                del x, g
                torch.cuda.empty_cache()
                continue

            # inputs read + fp8 out; bwd writes a 2N-wide grad
            out_n = 2 * N if bwd else N
            nbytes = M * 2 * N * 2 + (M * N * 2 if bwd else 0) + M * out_n
            manifest.append({"range": rng, "combo": combo, "dir": "na", "M": M, "N": N,
                             "swizzle": False, "bytes": nbytes, "iters": args.iters})
            print(f"WORKLOAD {rng} bytes={nbytes} iters={args.iters}", flush=True)

            for _ in range(args.warmup):
                fn()
            torch.cuda.synchronize()
            for _ in range(args.iters):
                if evict is not None:
                    evict.zero_()
                torch.cuda.synchronize()
                nvtx.range_push(rng)
                fn()
                torch.cuda.synchronize()
                nvtx.range_pop()

            del x, g
            torch.cuda.empty_cache()

    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump({"provenance": prov, "workloads": manifest}, f, indent=2)
        print(f"\nmanifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
