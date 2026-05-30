"""Benchmark the fused hybrid MXFP8+NVFP4 CuTeDSL kernel vs running TE's
MXFP8 and NVFP4 quantizers SEPARATELY (the baseline reads the input twice).

The hybrid kernel emits MXFP8 in one direction and NVFP4 in the other from a
SINGLE shared-memory read of the input tile. The baseline does the same two
quantizations as two independent TE calls (`tex.quantize` with an MXFP8Quantizer
then an NVFP4Quantizer), so it reads the input from HBM twice and launches two
kernels. The hybrid should win by fusing them into one read / one kernel.

Caveats (this is a kernel-fusion comparison, intentionally favorable):
  * The hybrid uses a precomputed global encode scale `s_enc` (passed in), so
    its timing excludes the global-amax pass. TE's NVFP4 `tex.quantize`
    computes its own amax internally — so the baseline includes that work.
  * SOL bytes are updated for the extra NVFP4 outputs: per call the hybrid
    moves  input(read once) + MXFP8 data + MXFP8 scale + NVFP4 data + NVFP4
    scale;  the separate baseline moves the input TWICE plus the same outputs.

Produces NVTX-tagged iterations for Nsight Systems (see run_nsys_profile.sh
--hybrid). Run directly for wall-clock timings.
"""

import argparse
import csv
import sys

import torch
import torch.cuda.nvtx as nvtx

import transformer_engine.pytorch as te  # loads libtransformer_engine.so first
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer, NVFP4Quantizer

import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack

from quantize_mxfp8_cutedsl_alt import (
    _hybrid_global_s_enc,
    _get_compiled_hybrid_kernel,
    MXFP8QuantizeConfig,
    _torch_to_cutlass_dtype,
)

# Reuse the shape presets + timing harness from the MXFP8 bench.
from bench_mxfp8_cutedsl import (
    SHAPE_PRESETS,
    parse_shapes,
    bench_once,
    _FP8_DTYPES,
    _TORCH_IN_DTYPES,
)

SCALE_DIM_MXFP8 = 32
SCALE_DIM_NVFP4 = 16


# ---------------------------------------------------------------------------
# Baseline: two SEPARATE TE quantizes (MXFP8 in one dir, NVFP4 in the other).
# ---------------------------------------------------------------------------
def make_baseline_fn(x, mxfp8_dir, fp8_dtype):
    """MXFP8 in `mxfp8_dir`, NVFP4 in the opposite direction — two TE calls,
    each reading the input from HBM independently."""
    mx_row = mxfp8_dir == "row"
    mxq = MXFP8Quantizer(
        fp8_dtype=_FP8_DTYPES[fp8_dtype],
        rowwise=mx_row, columnwise=not mx_row,
    )
    mxq.internal = True
    # NVFP4 in the OTHER direction. Plain NVFP4 (no RHT / 2D / stochastic) to
    # match the CuTeDSL kernel's math.
    nvq = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=not mx_row, columnwise=mx_row,
        with_rht=False, with_2d_quantization=False, stochastic_rounding=False,
    )
    nvq.internal = True

    def f():
        tex.quantize(x, mxq)
        tex.quantize(x, nvq)
    return f


# ---------------------------------------------------------------------------
# Hybrid: one fused CuTeDSL kernel. s_enc precomputed once (no per-call sync).
# ---------------------------------------------------------------------------
def make_hybrid_fn(x, mxfp8_dir, fp8_dtype):
    """Pre-build everything once (compile, output buffers, DLPack views, s_enc)
    and return a closure that just LAUNCHES the fused kernel. We measure the
    kernel, not the Python wrapper — kernel-only numbers come from nsys; the
    production path will dispatch this from C++."""
    M, N = x.shape
    mx_row = mxfp8_dir == "row"
    cfg = MXFP8QuantizeConfig(
        _torch_to_cutlass_dtype[x.dtype], fp8_dtype,
        rowwise=mx_row, colwise=not mx_row,
        nvfp4_rowwise=not mx_row, nvfp4_colwise=mx_row,
    )
    compiled = _get_compiled_hybrid_kernel(cfg, M, N)
    s_enc = _hybrid_global_s_enc(x)
    dev = x.device

    bufs = []  # keep output tensors alive for the closure's lifetime
    args = [from_dlpack(x, assumed_align=16)]

    def add(shape, enabled):
        if enabled:
            t = torch.empty(shape, dtype=torch.uint8, device=dev)
            bufs.append(t)
            args.append(from_dlpack(t, assumed_align=16))
        else:
            args.append(None)

    # Order must match HybridQuantizeSmemKernel.__call__.
    add((M, N), mx_row)                            # mO_row
    add((M, N // SCALE_DIM_MXFP8), mx_row)         # mS_row
    add((M, N), not mx_row)                        # mO_col
    add((M // SCALE_DIM_MXFP8, N), not mx_row)     # mS_col
    args.append(None)                              # mAmax
    add((M, N // 2), not mx_row)                   # mO_nvfp4_row
    add((M, N // SCALE_DIM_NVFP4), not mx_row)     # mS_nvfp4_row
    add((N, M // 2), mx_row)                       # mO_nvfp4_col (transposed)
    add((N, M // SCALE_DIM_NVFP4), mx_row)         # mS_nvfp4_col (transposed)
    args.append(Float32(s_enc))                    # s_enc

    def f():
        compiled(*args)
    return f


def _sol_bytes(M, N, in_elt):
    """Bytes moved per call. Hybrid reads input once; the separate baseline
    reads it twice. Outputs are identical between the two."""
    bytes_in = M * N * in_elt
    mxfp8_data = M * N                       # uint8 fp8
    mxfp8_scale = M * N // SCALE_DIM_MXFP8    # e8m0
    nvfp4_data = M * N // 2                   # 2 fp4 per byte
    nvfp4_scale = M * N // SCALE_DIM_NVFP4    # e4m3 per 16-block
    out_bytes = mxfp8_data + mxfp8_scale + nvfp4_data + nvfp4_scale
    hybrid_bytes = bytes_in + out_bytes
    baseline_bytes = 2 * bytes_in + out_bytes
    return hybrid_bytes, baseline_bytes


def bench_shape(M, N, mxfp8_dir, warmup, iters, fp8_dtype="e4m3",
                in_dtype="bf16", evict_l2=False, single=False):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    in_dt = _TORCH_IN_DTYPES[in_dtype]
    x = torch.randn(M, N, dtype=in_dt, device="cuda")

    nvfp4_dir = "col" if mxfp8_dir == "row" else "row"
    tag = f"hybrid_mx{mxfp8_dir}_nv{nvfp4_dir}_{in_dtype}_{fp8_dtype}"
    nvtx.range_push(f"shape_{M}x{N}_{mxfp8_dir}_{tag}")

    base_fn = make_baseline_fn(x, mxfp8_dir, fp8_dtype)
    hyb_fn = make_hybrid_fn(x, mxfp8_dir, fp8_dtype)

    # Warm the CuTeDSL JIT cache once (compile is not part of the measurement).
    nvtx.range_push("warm_jit")
    hyb_fn()
    torch.cuda.synchronize()
    nvtx.range_pop()

    ref_ms = bench_once(
        f"sep_te_{M}x{N}_{mxfp8_dir}_{tag}",
        base_fn, warmup, iters, evict_l2=evict_l2, single=single,
    )
    dsl_ms = bench_once(
        f"hybrid_{M}x{N}_{mxfp8_dir}_{tag}",
        hyb_fn, warmup, iters, evict_l2=evict_l2, single=single,
    )

    nvtx.range_pop()  # close shape_ range

    hybrid_bytes, baseline_bytes = _sol_bytes(M, N, x.element_size())
    return {
        "shape": (M, N),
        "dir": mxfp8_dir,
        "tag": tag,
        "ref_ms": ref_ms,
        "dsl_ms": dsl_ms,
        "ref_bw": baseline_bytes / (ref_ms * 1e-3) / 1e9,   # GB/s
        "dsl_bw": hybrid_bytes / (dsl_ms * 1e-3) / 1e9,     # GB/s
        "hybrid_bytes": hybrid_bytes,
        "baseline_bytes": baseline_bytes,
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--mxfp8-dir", choices=["row", "col", "all"], default="row",
                        help="MXFP8 direction; NVFP4 goes in the other one")
    # `--direction` is accepted as an alias so run_nsys_profile.sh can pass it
    # uniformly with the MXFP8 bench (row/col map to --mxfp8-dir).
    parser.add_argument("--direction", choices=["row", "col"], default=None,
                        help="alias for --mxfp8-dir (row|col)")
    parser.add_argument("--in-dtype", type=str, default="bf16",
                        choices=["bf16", "fp16"])
    parser.add_argument("--fp8", type=str, default="e4m3", choices=sorted(_FP8_DTYPES))
    parser.add_argument("--evict-l2", dest="evict_l2", action="store_true", default=True,
                        help="Flush L2 before each timed iter (default; cold cache).")
    parser.add_argument("--no-evict-l2", dest="evict_l2", action="store_false",
                        help="Warm-cache pipelined loop.")
    parser.add_argument("--single", action="store_true",
                        help="One-shot cold-cache measurement (overrides --iters).")
    parser.add_argument("--preset", type=str, default=None, choices=sorted(SHAPE_PRESETS))
    parser.add_argument("--shapes", type=str, default=None,
                        help="Custom shapes: 'M,N;M,N;...'  Overrides --preset")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if args.list_presets:
        for name, shapes in SHAPE_PRESETS.items():
            print(f"  {name:8s} " + ", ".join(f"{m}x{n}" for m, n in shapes))
        return 0

    if args.shapes:
        shapes = parse_shapes(args.shapes)
    elif args.preset:
        shapes = SHAPE_PRESETS[args.preset]
    else:
        shapes = SHAPE_PRESETS["default"]

    mxfp8_dir = args.direction or args.mxfp8_dir
    dirs = ["row", "col"] if mxfp8_dir == "all" else [mxfp8_dir]

    print(f"Hybrid bench: {len(shapes)} shape(s) × {len(dirs)} MXFP8-dir(s)  "
          f"in={args.in_dtype} fp8={args.fp8}")
    print("  baseline = TE MXFP8 + TE NVFP4 (separate, 2 input reads); "
          "DSL = fused hybrid (1 read)")
    print(f"  warmup={args.warmup} iters={args.iters} "
          f"evict_l2={args.evict_l2} single={args.single}")
    for m, n in shapes:
        print(f"  - {m}x{n}")
    print()

    torch.cuda.profiler.start()
    results = []
    for d in dirs:
        for M, N in shapes:
            results.append(bench_shape(
                M, N, d, args.warmup, args.iters, fp8_dtype=args.fp8,
                in_dtype=args.in_dtype, evict_l2=args.evict_l2, single=args.single))
    torch.cuda.profiler.stop()

    # Summary — column layout matches bench_mxfp8_cutedsl.py so the nsys
    # wrapper's stdout parser ($2=shape, $3=dir, $4=baseline_us, $5=dsl_us) works.
    print()
    print(f"{'tag':>34}  {'shape':>12}  {'dir':>4}  {'TE sep us':>9}  "
          f"{'hybrid us':>9}  {'TE GB/s':>9}  {'hyb GB/s':>9}  {'speedup':>7}")
    print("-" * 114)
    for r in results:
        M, N = r["shape"]
        speedup = r["ref_ms"] / r["dsl_ms"]
        print(f"{r['tag']:>34}  {M:6d}x{N:<5d}  {r['dir']:>4}  "
              f"{r['ref_ms']*1000:9.2f}  {r['dsl_ms']*1000:9.2f}  "
              f"{r['ref_bw']:9.1f}  {r['dsl_bw']:9.1f}  {speedup:6.2f}x")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tag", "M", "N", "mxfp8_dir", "te_sep_us", "hybrid_us",
                        "te_gbps", "hybrid_gbps", "speedup",
                        "hybrid_bytes", "baseline_bytes"])
            for r in results:
                M, N = r["shape"]
                w.writerow([r["tag"], M, N, r["dir"],
                            f"{r['ref_ms']*1000:.3f}", f"{r['dsl_ms']*1000:.3f}",
                            f"{r['ref_bw']:.2f}", f"{r['dsl_bw']:.2f}",
                            f"{r['ref_ms']/r['dsl_ms']:.3f}",
                            r["hybrid_bytes"], r["baseline_bytes"]])
        print(f"\nCSV written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
