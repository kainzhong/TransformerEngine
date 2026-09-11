# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""A/B driver for the rowwise MXFP8 remainder strategy.

The register-resident rowwise kernel (cast_rowwise.cu) covers a whole number of
MX blocks per CTA and hands the leftover blocks to a second, tiny kernel.  The
alternative is a single bounds-checked launch.  Only the *total* matters here --
a second launch costs GPU-side dispatch latency that a per-kernel duration sum
would never show -- so this times a long burst of back-to-back quantize calls
with one CUDA-event pair around the whole burst.

Inputs rotate through enough distinct buffers to exceed L2, so each call reads
cold data without an explicit flush kernel (a flush would leave the GPU idle
and let per-call launch latency leak into the measurement).

Run with cwd = repo root so the local libtransformer_engine.so is the one under
test.
"""

import argparse
import json
import os
import time

import torch
import torch.cuda.nvtx as nvtx

import transformer_engine.pytorch as te  # must precede transformer_engine_torch
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

# Mirrors cast_rowwise.cu: MX block size and the tier table that picks a launch
# configuration from the output size.
BLOCK_ELEMS = 32
TIER_MAX_BYTES = [12 << 20, 48 << 20, 96 << 20]
# (threads_per_cta, blocks_per_lane)
TIER_CONFIGS = [(256, 1), (256, 2), (128, 2), (256, 2)]


def tier_of(rows, cols):
    out_bytes = rows * cols
    tier = 0
    while tier < len(TIER_CONFIGS) - 1 and out_bytes > TIER_MAX_BYTES[tier]:
        tier += 1
    return tier


def blocks_per_cta(rows, cols):
    threads, bpl = TIER_CONFIGS[tier_of(rows, cols)]
    return threads // 2 * bpl


def remainder_blocks(rows, cols):
    """MX blocks the contiguous kernel leaves to the remainder kernel."""
    num_blocks = rows * (cols // BLOCK_ELEMS)
    return num_blocks % blocks_per_cta(rows, cols)


def mapped_lib():
    for line in open("/proc/self/maps"):
        if "libtransformer_engine" in line:
            return line.split()[-1]
    return "?"


def make_quantizer():
    q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    q.internal = True  # skip the Float8Tensor wrapper / python overhead
    return q


def time_shape(rows, cols, iters, warmup, reps, l2_bytes, tag=""):
    """Mean GPU time per quantize call, in microseconds."""
    q = make_quantizer()

    # Rotate over enough inputs that a call never reads data left in L2 by the
    # previous one.  Outputs rotate with them so write-back cannot alias either.
    in_bytes = rows * cols * 2
    nbuf = max(2, min(8, -(-(3 * l2_bytes) // in_bytes)))
    xs = [torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) for _ in range(nbuf)]
    outs = [tex.quantize(x, q) for x in xs]

    def burst(n):
        for i in range(n):
            j = i % nbuf
            tex.quantize(xs[j], q, outs[j], None)

    burst(warmup)
    torch.cuda.synchronize()

    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        nvtx.range_push(tag)
        t0 = time.perf_counter()
        burst(iters)
        enqueue_s = time.perf_counter() - t0
        nvtx.range_pop()
        end.record()
        torch.cuda.synchronize()
        samples.append((start.elapsed_time(end) * 1e3 / iters, enqueue_s * 1e6 / iters))

    samples.sort()
    gpu_us, cpu_us = samples[len(samples) // 2]
    return {
        "gpu_us": gpu_us,
        "cpu_enqueue_us": cpu_us,
        "min_us": samples[0][0],
        "max_us": samples[-1][0],
        "nbuf": nbuf,
    }


def checksum(rows, cols, seed=0):
    """Bitwise fingerprint of one quantize, for cross-arm correctness.

    The scale array is allocated with its row count rounded up to 128, and the
    kernel never writes the pad, so the fingerprint must skip it -- otherwise it
    reports uninitialized memory and differs run to run.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16, generator=g)
    out = tex.quantize(x, make_quantizer())
    data = out._rowwise_data.view(torch.uint8).reshape(rows, cols)
    scales = out._rowwise_scale_inv.view(torch.uint8)[:rows, : cols // BLOCK_ELEMS]
    # Sum in int64 so nothing wraps; enough to catch a missed or corrupted block.
    return [int(data.to(torch.int64).sum()), int(scales.to(torch.int64).sum())]


# The PyTorch entry point requires both tensor dims to be multiples of 32, and
# the specialized rowwise kernel additionally needs cols % 128 == 0.  Writing
# rows = 32a and cols = 128b, the block count is 128ab, so it is short of a full
# CTA only when a CTA covers 256 blocks (tiers 1 and 3) and a and b are both odd
# -- and then the leftover is always exactly 128 blocks.  Tiers 0 and 2 use
# 128-block CTAs and can never leave a remainder from this entry point.
#
# Each pair below differs by 32 rows, which flips the parity of a and so toggles
# the remainder while changing the work by well under a percent.  The even-row
# member is the control: the two arms must agree on it.
SHAPES = [
    # tier 1 (12 - 48 MiB out)
    (8160, 1920),   # a=255, b=15  -> remainder
    (8192, 1920),   # a=256        -> exact
    (4064, 4480),   # a=127, b=35  -> remainder
    (4096, 4480),   # a=128        -> exact
    # tier 3 (> 96 MiB out)
    (65504, 1920),  # a=2047, b=15 -> remainder
    (65536, 1920),  # a=2048       -> exact
    (32736, 4480),  # a=1023, b=35 -> remainder
    (32768, 4480),  # a=1024       -> exact
    # tier 2 (48 - 96 MiB): 128-block CTAs, so never a remainder
    (16384, 4480),
    # shapes that occur in practice, all exact; these should not move at all
    (4096, 4096),
    (8192, 14336),
    (32768, 4096),
    (2048, 2048),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, help="label for this build")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--out", default=None)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()

    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    print(f"arm={args.arm} lib={mapped_lib()} L2={l2_bytes/2**20:.0f} MiB")

    results = {"arm": args.arm, "lib": mapped_lib(), "shapes": {}}
    for rows, cols in SHAPES:
        key = f"{rows}x{cols}"
        rem = remainder_blocks(rows, cols)
        num_blocks = rows * (cols // BLOCK_ELEMS)
        grid = num_blocks // blocks_per_cta(rows, cols)
        entry = {
            "rows": rows,
            "cols": cols,
            "tier": tier_of(rows, cols),
            "remainder_blocks": rem,
            "contiguous_ctas": grid,
            "checksum": checksum(rows, cols),
        }
        if not args.check_only:
            entry.update(
                time_shape(
                    rows, cols, args.iters, args.warmup, args.reps, l2_bytes, tag=f"RB|{key}"
                )
            )
            print(
                f"{key:>14}  tier{entry['tier']}  rem={rem:>4}  ctas={grid:>6}  "
                f"gpu={entry['gpu_us']:8.2f} us  (cpu enqueue {entry['cpu_enqueue_us']:6.2f} us)"
            )
        else:
            print(f"{key:>14}  rem={rem:>4}  checksum={entry['checksum']}")
        results["shapes"][key] = entry

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
