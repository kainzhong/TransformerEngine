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

from quantize_mxfp8_cutedsl_alt import get_quantize_mxfp8_cutedsl_func

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

# combo → dsl_activation_kwarg
COMBOS = {
    # Plain quantize — matches MXFP8Quantizer(...).__call__.
    "plain":  None,
    # Forward fused activation — tex.<name>.
    "gelu":   "gelu",
    "silu":   "silu",
    "relu":   "relu",
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


def make_reference_fn(combo, x, rowwise, colwise,
                      fp8_dtype="e4m3", swizzle=False, with_amax=False):
    """Return a 0-arg callable that invokes the C++ TE reference for `combo`."""
    quantizer = MXFP8Quantizer(
        fp8_dtype=_FP8_DTYPES[fp8_dtype],
        rowwise=rowwise,
        columnwise=colwise,
    )
    quantizer.internal = True
    if swizzle:
        quantizer.optimize_for_gemm = True
    # TE's MXFP8Quantizer doesn't expose a per-tensor amax knob via Python
    # (only via the C++ tensor.amax pointer). The DSL path-with-amax is
    # benchmarked against TE's plain path; the extra DSL work is the warp
    # redux + atomic. Caller passes with_amax=True only on the DSL side.
    #
    # Use the direct `tex.quantize(x, quantizer)` pybind for plain so the C++
    # entry point's Python overhead matches `tex.{relu,gelu,silu,...}` for
    # activation combos. The alternative `quantizer(x)` goes through
    # `MXFP8Quantizer.__call__` which wraps the result in `Float8Tensor` and
    # adds ~15 us of Python overhead — making wall-clock comparisons across
    # combos misleading.
    if combo == "plain":
        return lambda: tex.quantize(x, quantizer)
    if combo in _TEX_FORWARD_ACT:
        op = _TEX_FORWARD_ACT[combo]
        return lambda: op(x, quantizer)
    raise ValueError(f"unknown combo {combo!r}")

timing_dict = {}
timing_count_dict = {}
# Skip warm_jit (1 dsl call) + bench_once's own warmup (`warmup` dsl calls).
skip_calls = 10
def timing_func(label, iter_ms):
    timing_count_dict[label] = timing_count_dict.get(label, 0) + 1
    if timing_count_dict[label] > skip_calls:
        timing_dict[label] = timing_dict.get(label, 0.0) + iter_ms

def make_dsl_fn(combo, x, rowwise, colwise,
                fp8_dtype="e4m3", swizzle=False, with_amax=False):
    """Return a 0-arg callable that invokes the CuTeDSL kernel for `combo`."""
    activation = COMBOS[combo]
    quantizer = MXFP8Quantizer(
        fp8_dtype=_FP8_DTYPES[fp8_dtype],
        rowwise=rowwise,
        columnwise=colwise,
    )
    quantizer.internal = True
    if swizzle:
        quantizer.optimize_for_gemm = True
    # quantize_with_func is gone (replaced by applyTVMFunction); for this
    # JIT bench we just need an empty output buffer, which is what
    # create_empty_quantized_tensor produces directly.
    fn_name = get_quantize_mxfp8_cutedsl_func(
        x=x,
        fp8_dtype="e4m3" if quantizer.dtype == tex.DType.kFloat8E4M3 else "e5m2",
        rowwise=quantizer.rowwise_usage,
        colwise=quantizer.columnwise_usage,
        with_gemm_swizzled_scales=quantizer.optimize_for_gemm,
        activation=None,
    )

    def combined():
        return tex.quantize_with_func(x, quantizer, None, fn_name)

    def separate():
        out = tex.prepare_quantize(x, quantizer)
        return tex.quantize_with_func(x, quantizer, out, fn_name)
    
    print("combined")
    return combined
    # print("separate")
    # return separate
    

# Module-level L2 evict buffer. 256 MB f32 (covers B200's ~60 MB L2 with headroom).
# Allocated lazily, reused across calls to avoid alloc churn between bench runs.
_L2_EVICT_BUF = None


def _l2_evict_buf():
    global _L2_EVICT_BUF
    if _L2_EVICT_BUF is None:
        _L2_EVICT_BUF = torch.empty(
            256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    return _L2_EVICT_BUF


def bench_once(name, fn, warmup, iters, evict_l2=False, single=False):
    """Time `iters` calls to `fn()` after `warmup` calls, wrapped in an NVTX range.

    Modes:
      - default: warm-cache, one event pair around the iter loop.
        iter via a 256MB write, times each iter with its own event pair, returns
        the average.
      - single=True: cold-cache one-shot — warmup with no eviction, then a
        single L2 flush, then exactly one timed kernel launch. Returns that
        single sample. `iters` is ignored.

    `single` takes precedence over `evict_l2` if both are set."""

    if single:
        # Warmup with no eviction — let the kernel cache (icache, JIT, etc.)
        # warm up without disturbing L2.
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        evict = _l2_evict_buf()
        nvtx.range_push(f"{name}_single")
        evict.zero_()             # async L2 flush
        torch.cuda.synchronize()  # ← drain evict; L2 is now cold AND GPU idle
        # Host wall-clock from here includes Python wrapper + kernel time;
        # excludes the evict kernel (already drained above).
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        nvtx.range_pop()
        return (t1 - t0) / 1e6    # ms (match cuda.Event.elapsed_time units)

    if evict_l2:
        # Warmup with no eviction (we're not timing warmup, no need to flush).
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        evict = _l2_evict_buf()
        total_ms = 0.0
        nvtx.range_push(f"{name}_measure")
        for i in range(iters):
            evict.zero_()             # async L2 flush
            torch.cuda.synchronize()  # ← drain evict before timing starts
            nvtx.range_push(f"{name}_iter_{i}")
            t0 = time.perf_counter_ns()
            fn()                      # host wrapper + kernel launch
            torch.cuda.synchronize()  # wait for kernel
            t1 = time.perf_counter_ns()
            nvtx.range_pop()
            total_ms += (t1 - t0) / 1e6
        nvtx.range_pop()
        return total_ms / iters

    # Default (warm-cache) path — one event pair across the whole loop.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    total_ms = 0.0
    t0 = time.perf_counter_ns()
    for i in range(iters):
        # evict.zero_()             # async L2 flush
        # torch.cuda.synchronize()  # ← drain evict before timing starts
        # nvtx.range_push(f"{name}_iter_{i}")
        # t0 = time.perf_counter_ns()
        fn()                      # host wrapper + kernel launch
        # torch.cuda.synchronize()  # wait for kernel
        # t1 = time.perf_counter_ns()
        # nvtx.range_pop()
        # total_ms += (t1 - t0) / 1e6
    t1 = time.perf_counter_ns()
    total_ms = (t1 - t0) / 1e6
    return total_ms / iters


def bench_shape(M, N, rowwise, colwise, warmup, iters, combo="plain",
                in_dtype="bf16", fp8_dtype="e4m3", swizzle=False,
                with_amax=False, evict_l2=False, single=False):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    in_dt = _TORCH_IN_DTYPES[in_dtype]
    x = torch.randn(M, N, dtype=in_dt, device="cuda")

    dir_label = "both" if (rowwise and colwise) else ("row" if rowwise else "col")
    tag = f"{combo}_{in_dtype}_{fp8_dtype}"
    if swizzle:
        tag += "_sw"
    if with_amax:
        tag += "_am"

    # Per-shape outer NVTX range so each shape is clearly grouped in the timeline
    nvtx.range_push(f"shape_{M}x{N}_{dir_label}_{tag}")

    ref_fn = make_reference_fn(combo, x, rowwise, colwise,
                               fp8_dtype=fp8_dtype, swizzle=swizzle,
                               with_amax=with_amax)
    dsl_fn = make_dsl_fn(combo, x, rowwise, colwise,
                         fp8_dtype=fp8_dtype, swizzle=swizzle,
                         with_amax=with_amax)
    # Warm the CuTeDSL JIT cache once (not counted against bench)
    nvtx.range_push("warm_jit")
    dsl_fn()
    torch.cuda.synchronize()
    nvtx.range_pop()

    ref_ms = bench_once(
        f"cpp_ref_{M}x{N}_{dir_label}_{tag}",
        ref_fn, warmup, iters, evict_l2=evict_l2, single=single,
    )

    dsl_ms = bench_once(
        f"cutedsl_{M}x{N}_{dir_label}_{tag}",
        dsl_fn, warmup, iters, evict_l2=evict_l2, single=single,
    )

    # timing_func is invoked once per label per dsl_fn() call, so the number of
    # dsl iterations is timing_calls / len(timing_dict).
    for k, v in timing_dict.items():
        v = v / iters if (not single and iters) else v
        v = v * 1000  # convert to us for readability
        print(f"  timing_func: {k} = {v:.3f} us")

    nvtx.range_pop()  # close shape_ range

    in_bytes_per_elt = x.element_size()
    bytes_in = M * N * in_bytes_per_elt
    bytes_out = 0
    bytes_scale = 0
    if rowwise:
        bytes_out += M * N * 1                # rowwise FP8 data (uint8)
        bytes_scale += M * (N // 32)          # rowwise e8m0 scales
    if colwise:
        bytes_out += M * N * 1                # colwise FP8 data
        bytes_scale += (M // 32) * N          # colwise e8m0 scales
    total_bytes = bytes_in + bytes_out + bytes_scale

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
    # L2 eviction is ON by default so the bench reflects a production cold-cache
    # call (one kernel launch's worth of input being read from HBM, not L2).
    # Pass --no-evict-l2 to disable for a warm-cache pipelined measurement.
    parser.add_argument("--evict-l2", dest="evict_l2", action="store_true",
                        default=True,
                        help="Flush L2 cache before each timed iter "
                             "(default; cold-cache measurement). Wrapper Python "
                             "overhead is included in the timing.")
    parser.add_argument("--no-evict-l2", dest="evict_l2", action="store_false",
                        help="Disable L2 eviction; run a warm-cache pipelined "
                             "loop instead (legacy behavior).")
    parser.add_argument("--single", action="store_true",
                        help="One-shot cold-cache measurement: warmup, flush L2 "
                             "once, time a single kernel launch, report that "
                             "sample. Overrides --iters. Takes precedence over "
                             "--evict-l2.")
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
                                        with_amax=args.with_amax,
                                        evict_l2=args.evict_l2,
                                        single=args.single)
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
