"""Benchmark CuTeDSL MXFP8 quantization vs. C++ reference.

Produces NVTX-tagged iterations for Nsight Systems timeline profiling.
Run directly for wall-clock timings, or via run_nsys_profile.sh for nsys capture.
"""

import argparse
import time

import torch
import torch.cuda.nvtx as nvtx

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

from quantize_mxfp8_cutedsl import quantize_mxfp8_cutedsl


DEFAULT_SHAPES = [
    (1024, 1024),
    (4096, 4096),
    (8192, 8192),
    (16384, 8192),
    (16384, 16384),
]


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

    # Warm the CuTeDSL JIT cache once (not counted against bench)
    nvtx.range_push(f"shape_{M}x{N}_warm_jit")
    quantize_mxfp8_cutedsl(x, rowwise=rowwise, colwise=colwise)
    torch.cuda.synchronize()
    nvtx.range_pop()

    # Direction label
    dir_label = "both" if (rowwise and colwise) else ("row" if rowwise else "col")

    # Reference
    ref_ms = bench_once(
        f"cpp_ref_{M}x{N}_{dir_label}",
        lambda: reference_quantize(x, rowwise, colwise),
        warmup, iters,
    )

    # CuTeDSL
    dsl_ms = bench_once(
        f"cutedsl_{M}x{N}_{dir_label}",
        lambda: quantize_mxfp8_cutedsl(x, rowwise=rowwise, colwise=colwise),
        warmup, iters,
    )

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--direction", choices=["row", "col", "both", "all"],
                        default="all",
                        help="Which direction(s) to benchmark")
    parser.add_argument("--shapes", type=str, default=None,
                        help="Comma-separated M,N pairs; e.g. '1024,1024;4096,4096'")
    args = parser.parse_args()

    if args.shapes:
        shapes = []
        for pair in args.shapes.split(";"):
            m, n = pair.split(",")
            shapes.append((int(m), int(n)))
    else:
        shapes = DEFAULT_SHAPES

    if args.direction == "all":
        dirs = [("row", True, False), ("col", False, True), ("both", True, True)]
    elif args.direction == "row":
        dirs = [("row", True, False)]
    elif args.direction == "col":
        dirs = [("col", False, True)]
    else:
        dirs = [("both", True, True)]

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
    print(f"{'shape':>12}  {'dir':>4}  {'C++ ms':>9}  {'DSL ms':>9}  "
          f"{'C++ GB/s':>9}  {'DSL GB/s':>9}  {'DSL/C++':>7}")
    print("-" * 80)
    for r in results:
        M, N = r["shape"]
        speedup = r["ref_ms"] / r["dsl_ms"]
        print(f"{M:6d}x{N:<5d}  {r['dir']:>4}  "
              f"{r['ref_ms']:9.4f}  {r['dsl_ms']:9.4f}  "
              f"{r['ref_bw']:9.1f}  {r['dsl_bw']:9.1f}  {speedup:6.2f}x")


if __name__ == "__main__":
    main()
