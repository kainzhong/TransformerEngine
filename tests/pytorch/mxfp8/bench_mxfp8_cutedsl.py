"""Benchmark CuTeDSL MXFP8 quantization vs. C++ reference.

Produces NVTX-tagged iterations for Nsight Systems timeline profiling.
Run directly for wall-clock timings, or via run_nsys_profile.sh for nsys capture.
"""

import argparse
import csv
import sys
import time

import torch
import torch.cuda.nvtx as nvtx

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

from quantize_mxfp8_cutedsl import quantize_mxfp8_cutedsl


# Shape presets — names map to lists of (M, N) pairs.
# All shapes are multiples of 64 (the CuTeDSL kernel's CHUNK_DIM).
SHAPE_PRESETS = {
    "tiny":   [(128, 128), (256, 256), (512, 512)],
    "small":  [(1024, 1024), (2048, 2048), (4096, 4096)],
    "medium": [(8192, 8192), (8192, 4096), (4096, 8192)],
    "large":  [(16384, 8192), (16384, 16384), (32768, 8192)],
    "square": [(1024, 1024), (2048, 2048), (4096, 4096),
               (8192, 8192), (16384, 16384)],
    # LLM-typical shapes: (batch*seq, hidden) for common hidden sizes
    "llm":    [(2048, 5120), (2048, 8192), (4096, 12288),
               (8192, 14336), (16384, 16384)],
    # Aspect-ratio sweep: tall-narrow and short-wide
    "aspect": [(1024, 16384), (4096, 4096), (16384, 1024),
               (512, 32768), (32768, 512)],
    "default": [(1024, 1024), (4096, 4096), (8192, 8192),
                (16384, 8192), (16384, 16384)],
    "test": [(2048, 5120)]
}


def parse_shapes(shapes_str: str):
    """Parse a ';'-separated list of 'M,N' pairs."""
    shapes = []
    for pair in shapes_str.split(";"):
        m, n = pair.strip().split(",")
        shapes.append((int(m), int(n)))
    return shapes


def reference_quantize(x, rowwise, colwise):
    """C++ kernel path via TE's MXFP8Quantizer."""
    q = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=colwise,
    )
    return q(x)


def bench_once(name, fn, warmup, iters):
    """Time `iters` calls to `fn()` after `warmup` calls, wrapped in an NVTX range."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    nvtx.range_push(f"{name}_measure")
    start.record()
    for i in range(iters):
        nvtx.range_push(f"{name}_iter_{i}")
        fn()
        nvtx.range_pop()
    end.record()
    nvtx.range_pop()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return total_ms / iters


def bench_shape(M, N, rowwise, colwise, warmup, iters):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    dir_label = "both" if (rowwise and colwise) else ("row" if rowwise else "col")

    # Per-shape outer NVTX range so each shape is clearly grouped in the timeline
    nvtx.range_push(f"shape_{M}x{N}_{dir_label}")

    # Warm the CuTeDSL JIT cache once (not counted against bench)
    nvtx.range_push("warm_jit")
    quantize_mxfp8_cutedsl(x, rowwise=rowwise, colwise=colwise)
    torch.cuda.synchronize()
    nvtx.range_pop()

    ref_ms = bench_once(
        f"cpp_ref_{M}x{N}_{dir_label}",
        lambda: reference_quantize(x, rowwise, colwise),
        warmup, iters,
    )

    dsl_ms = bench_once(
        f"cutedsl_{M}x{N}_{dir_label}",
        lambda: quantize_mxfp8_cutedsl(x, rowwise=rowwise, colwise=colwise),
        warmup, iters,
    )

    nvtx.range_pop()  # close shape_ range

    bytes_in = x.numel() * x.element_size()
    bytes_out = M * N * 1  # FP8 is 1 byte
    bytes_scale = (M * (N // 32)) if rowwise else 0
    bytes_scale += ((M // 32) * N) if colwise else 0
    if rowwise and colwise:
        bytes_out *= 2
    total_bytes = bytes_in + bytes_out + bytes_scale

    def bw(ms):
        return total_bytes / (ms * 1e-3) / 1e9  # GB/s

    return {
        "shape": (M, N),
        "dir": dir_label,
        "ref_ms": ref_ms,
        "dsl_ms": dsl_ms,
        "ref_bw": bw(ref_ms),
        "dsl_bw": bw(dsl_ms),
        "bytes": total_bytes,
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=(
            "Shape selection (pick one of --preset / --shapes):\n"
            "  --preset PRESET    named sweep (see --list-presets)\n"
            "  --shapes 'M,N;...' custom list\n"
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--direction", choices=["row", "col", "both", "all"],
                        default="all",
                        help="Which direction(s) to benchmark")
    parser.add_argument("--preset", type=str, default=None,
                        choices=sorted(SHAPE_PRESETS),
                        help=f"Shape preset: one of {sorted(SHAPE_PRESETS)}")
    parser.add_argument("--shapes", type=str, default=None,
                        help="Custom shapes: 'M,N;M,N;...'  Overrides --preset")
    parser.add_argument("--list-presets", action="store_true",
                        help="Print all presets and exit")
    parser.add_argument("--csv", type=str, default=None,
                        help="Write results as CSV to this file")
    args = parser.parse_args()

    if args.list_presets:
        for name, shapes in SHAPE_PRESETS.items():
            shapes_str = ", ".join(f"{m}x{n}" for m, n in shapes)
            print(f"  {name:8s} {shapes_str}")
        return 0

    if args.shapes:
        shapes = parse_shapes(args.shapes)
    elif args.preset:
        shapes = SHAPE_PRESETS[args.preset]
    else:
        shapes = SHAPE_PRESETS["default"]

    if args.direction == "all":
        dirs = [("row", True, False), ("col", False, True), ("both", True, True)]
    elif args.direction == "row":
        dirs = [("row", True, False)]
    elif args.direction == "col":
        dirs = [("col", False, True)]
    else:
        dirs = [("both", True, True)]

    print(f"Benchmarking {len(shapes)} shape(s) × {len(dirs)} direction(s)")
    print(f"  warmup={args.warmup} iters={args.iters}")
    for m, n in shapes:
        print(f"  - {m}x{n}")
    print()

    # Signal nsys to start capturing (used with --capture-range=cudaProfilerApi)
    torch.cuda.profiler.start()

    results = []
    for M, N in shapes:
        for _, rw, cw in dirs:
            r = bench_shape(M, N, rw, cw, args.warmup, args.iters)
            results.append(r)

    torch.cuda.profiler.stop()

    # Print summary
    print()
    print(f"{'shape':>12}  {'dir':>4}  {'C++ us':>9}  {'DSL us':>9}  "
          f"{'C++ GB/s':>9}  {'DSL GB/s':>9}  {'DSL/C++':>7}")
    print("-" * 80)
    for r in results:
        M, N = r["shape"]
        speedup = r["ref_ms"] / r["dsl_ms"]
        print(f"{M:6d}x{N:<5d}  {r['dir']:>4}  "
              f"{r['ref_ms']*1000:9.2f}  {r['dsl_ms']*1000:9.2f}  "
              f"{r['ref_bw']:9.1f}  {r['dsl_bw']:9.1f}  {speedup:6.2f}x")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["M", "N", "dir", "ref_us", "dsl_us",
                        "ref_gbps", "dsl_gbps", "speedup", "bytes"])
            for r in results:
                M, N = r["shape"]
                w.writerow([M, N, r["dir"],
                            f"{r['ref_ms']*1000:.3f}",
                            f"{r['dsl_ms']*1000:.3f}",
                            f"{r['ref_bw']:.2f}",
                            f"{r['dsl_bw']:.2f}",
                            f"{r['ref_ms']/r['dsl_ms']:.3f}",
                            r["bytes"]])
        print(f"\nCSV written to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
