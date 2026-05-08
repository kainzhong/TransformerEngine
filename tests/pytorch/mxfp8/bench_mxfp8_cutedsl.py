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

from quantize_mxfp8_cutedsl_alt import quantize_mxfp8_cutedsl


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


_TEX_FORWARD_ACT = {"gelu": tex.gelu, "silu": tex.silu, "relu": tex.relu}
_TEX_BACKWARD_ACT = {"dgelu": tex.dgelu, "dsilu": tex.dsilu, "drelu": tex.drelu}
_TEX_DBIAS_DACT = {
    "dbias_dgelu": tex.dbias_dgelu,
    "dbias_dsilu": tex.dbias_dsilu,
    "dbias_drelu": tex.dbias_drelu,
}

# combo → (needs_act_input, dsl_activation_kwarg, dsl_compute_dbias)
COMBOS = {
    # Plain quantize — matches MXFP8Quantizer(...).__call__.
    "plain":         (False, None,    False),
    # Forward fused activation — tex.<name>.
    "gelu":          (False, "gelu",  False),
    "silu":          (False, "silu",  False),
    "relu":          (False, "relu",  False),
    # Backward activation only — tex.<name>(grad, act_in, q).
    "dgelu":         (True,  "dgelu", False),
    "dsilu":         (True,  "dsilu", False),
    "drelu":         (True,  "drelu", False),
    # Backward activation + bias gradient — tex.dbias_<name>(grad, act_in, q).
    "dbias_dgelu":   (True,  "dgelu", True),
    "dbias_dsilu":   (True,  "dsilu", True),
    "dbias_drelu":   (True,  "drelu", True),
}


_FP8_DTYPES = {
    "e4m3": tex.DType.kFloat8E4M3,
    "e5m2": tex.DType.kFloat8E5M2,
}
_TORCH_IN_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def make_reference_fn(combo, x, act_in, rowwise, colwise,
                      fp8_dtype="e4m3", swizzle=False, with_amax=False):
    """Return a 0-arg callable that invokes the C++ TE reference for `combo`."""
    quantizer = MXFP8Quantizer(
        fp8_dtype=_FP8_DTYPES[fp8_dtype],
        rowwise=rowwise,
        columnwise=colwise,
    )
    if swizzle:
        quantizer.optimize_for_gemm = True
    # TE's MXFP8Quantizer doesn't expose a per-tensor amax knob via Python
    # (only via the C++ tensor.amax pointer). The DSL path-with-amax is
    # benchmarked against TE's plain path; the extra DSL work is the warp
    # redux + atomic. Caller passes with_amax=True only on the DSL side.
    if combo == "plain":
        return lambda: quantizer(x)
    if combo in _TEX_FORWARD_ACT:
        op = _TEX_FORWARD_ACT[combo]
        return lambda: op(x, quantizer)
    if combo in _TEX_BACKWARD_ACT:
        op = _TEX_BACKWARD_ACT[combo]
        return lambda: op(x, act_in, quantizer)
    if combo in _TEX_DBIAS_DACT:
        op = _TEX_DBIAS_DACT[combo]
        return lambda: op(x, act_in, quantizer)
    raise ValueError(f"unknown combo {combo!r}")


def make_dsl_fn(combo, x, act_in, rowwise, colwise,
                fp8_dtype="e4m3", swizzle=False, amax=None):
    """Return a 0-arg callable that invokes the CuTeDSL kernel for `combo`."""
    _, activation, compute_dbias = COMBOS[combo]
    kwargs = dict(rowwise=rowwise, colwise=colwise,
                  fp8_dtype=fp8_dtype,
                  with_gemm_swizzled_scales=swizzle,
                  amax=amax)
    if activation is None:
        return lambda: quantize_mxfp8_cutedsl(x, **kwargs)
    kwargs.update(activation=activation, act_input=act_in,
                  compute_dbias=compute_dbias)
    return lambda: quantize_mxfp8_cutedsl(x, **kwargs)


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


def bench_shape(M, N, rowwise, colwise, warmup, iters, combo="plain",
                in_dtype="bf16", fp8_dtype="e4m3", swizzle=False,
                with_amax=False):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    in_dt = _TORCH_IN_DTYPES[in_dtype]
    x = torch.randn(M, N, dtype=in_dt, device="cuda")
    needs_act_input, _, has_dbias = COMBOS[combo]
    act_in = (torch.randn(M, N, dtype=in_dt, device="cuda")
              if needs_act_input else None)
    amax_buf = (torch.zeros(1, dtype=torch.float32, device="cuda")
                if with_amax else None)

    dir_label = "both" if (rowwise and colwise) else ("row" if rowwise else "col")
    tag = f"{combo}_{in_dtype}_{fp8_dtype}"
    if swizzle:
        tag += "_sw"
    if with_amax:
        tag += "_am"

    # Per-shape outer NVTX range so each shape is clearly grouped in the timeline
    nvtx.range_push(f"shape_{M}x{N}_{dir_label}_{tag}")

    ref_fn = make_reference_fn(combo, x, act_in, rowwise, colwise,
                               fp8_dtype=fp8_dtype, swizzle=swizzle,
                               with_amax=with_amax)
    dsl_fn = make_dsl_fn(combo, x, act_in, rowwise, colwise,
                         fp8_dtype=fp8_dtype, swizzle=swizzle,
                         amax=amax_buf)

    # Warm the CuTeDSL JIT cache once (not counted against bench)
    nvtx.range_push("warm_jit")
    dsl_fn()
    torch.cuda.synchronize()
    nvtx.range_pop()

    ref_ms = bench_once(
        f"cpp_ref_{M}x{N}_{dir_label}_{tag}",
        ref_fn, warmup, iters,
    )

    dsl_ms = bench_once(
        f"cutedsl_{M}x{N}_{dir_label}_{tag}",
        dsl_fn, warmup, iters,
    )

    nvtx.range_pop()  # close shape_ range

    in_bytes_per_elt = x.element_size()
    bytes_in = M * N * in_bytes_per_elt
    if needs_act_input:
        bytes_in += M * N * in_bytes_per_elt
    bytes_out = 0
    bytes_scale = 0
    if rowwise:
        bytes_out += M * N * 1                # rowwise FP8 data (uint8)
        bytes_scale += M * (N // 32)          # rowwise e8m0 scales
    if colwise:
        bytes_out += M * N * 1                # colwise FP8 data
        bytes_scale += (M // 32) * N          # colwise e8m0 scales
    bytes_dbias = 0
    if has_dbias:
        # Approximate: workspace = blocks_Y · N · 4 (f32). The reduce step
        # reads it back and writes a tiny dbias[N] in input dtype.
        blocks_Y = (M + 63) // 64
        bytes_dbias = blocks_Y * N * 4 + N * in_bytes_per_elt
    total_bytes = bytes_in + bytes_out + bytes_scale + bytes_dbias

    def bw(ms):
        return total_bytes / (ms * 1e-3) / 1e9  # GB/s

    return {
        "shape": (M, N),
        "dir": dir_label,
        "combo": combo,
        "tag": tag,
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
    parser.add_argument("--combo", type=str, default="plain",
                        choices=sorted(COMBOS),
                        help=f"Operation: one of {sorted(COMBOS)}. "
                             "Use 'all' to sweep multiple at once.")
    parser.add_argument("--combos", type=str, default=None,
                        help="Comma-separated list of combos (overrides --combo)")
    parser.add_argument("--in-dtype", type=str, default="bf16",
                        choices=sorted(_TORCH_IN_DTYPES))
    parser.add_argument("--in-dtypes", type=str, default=None,
                        help="Comma-separated list of input dtypes")
    parser.add_argument("--fp8", type=str, default="e4m3",
                        choices=sorted(_FP8_DTYPES))
    parser.add_argument("--fp8s", type=str, default=None,
                        help="Comma-separated list of fp8 output dtypes")
    parser.add_argument("--swizzle", action="store_true",
                        help="Enable WITH_GEMM_SWIZZLED_SCALES")
    parser.add_argument("--with-amax", action="store_true",
                        help="Enable per-tensor amax accumulation (DSL only)")
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

    if args.combos:
        combos = [c.strip() for c in args.combos.split(",")]
        for c in combos:
            if c not in COMBOS:
                print(f"unknown combo: {c}", file=sys.stderr)
                return 1
    else:
        combos = [args.combo]

    in_dtypes = ([d.strip() for d in args.in_dtypes.split(",")]
                 if args.in_dtypes else [args.in_dtype])
    fp8s = ([d.strip() for d in args.fp8s.split(",")]
            if args.fp8s else [args.fp8])

    print(f"Benchmarking {len(shapes)} shape(s) × {len(dirs)} direction(s) × "
          f"{len(combos)} combo(s) × {len(in_dtypes)} in-dtype × "
          f"{len(fp8s)} fp8")
    print(f"  warmup={args.warmup} iters={args.iters}")
    print(f"  combos: {combos}  in_dtypes: {in_dtypes}  fp8: {fp8s}  "
          f"swizzle={args.swizzle}  with_amax={args.with_amax}")
    for m, n in shapes:
        print(f"  - {m}x{n}")
    print()

    # Signal nsys to start capturing (used with --capture-range=cudaProfilerApi)
    torch.cuda.profiler.start()

    results = []
    for combo in combos:
        for in_dtype in in_dtypes:
            for fp8 in fp8s:
                for M, N in shapes:
                    for _, rw, cw in dirs:
                        r = bench_shape(M, N, rw, cw, args.warmup, args.iters,
                                        combo, in_dtype=in_dtype, fp8_dtype=fp8,
                                        swizzle=args.swizzle,
                                        with_amax=args.with_amax)
                        results.append(r)

    torch.cuda.profiler.stop()

    # Print summary
    print()
    print(f"{'tag':>34}  {'shape':>12}  {'dir':>4}  {'C++ us':>9}  {'DSL us':>9}  "
          f"{'C++ GB/s':>9}  {'DSL GB/s':>9}  {'DSL/C++':>7}")
    print("-" * 110)
    for r in results:
        M, N = r["shape"]
        speedup = r["ref_ms"] / r["dsl_ms"]
        print(f"{r['tag']:>34}  {M:6d}x{N:<5d}  {r['dir']:>4}  "
              f"{r['ref_ms']*1000:9.2f}  {r['dsl_ms']*1000:9.2f}  "
              f"{r['ref_bw']:9.1f}  {r['dsl_bw']:9.1f}  {speedup:6.2f}x")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tag", "combo", "M", "N", "dir", "ref_us", "dsl_us",
                        "ref_gbps", "dsl_gbps", "speedup", "bytes"])
            for r in results:
                M, N = r["shape"]
                w.writerow([r["tag"], r["combo"], M, N, r["dir"],
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
