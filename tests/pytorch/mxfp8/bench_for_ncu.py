"""NCU-friendly benchmark: one (shape, direction) per invocation.

Profiles ONE C++ reference launch and ONE CuTeDSL launch within a single
Python process.  Warmup runs outside torch.cuda.profiler.start/stop, so
ncu (invoked with --profile-from-start off) only captures the two
measurement launches — one per kernel.
"""

import argparse
import sys

import torch
import torch.cuda.nvtx as nvtx

import transformer_engine.pytorch as te  # noqa: F401  (import registers the extension)
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

from quantize_mxfp8_cutedsl_alt import quantize_mxfp8_cutedsl
from bench_mxfp8_cutedsl import parse_shapes


# Module-level L2 evict buffer (256 MB f32, covers B200's ~60 MB L2 with headroom).
# Allocated lazily, reused across calls.
_L2_EVICT_BUF = None


def _l2_evict_buf():
    global _L2_EVICT_BUF
    if _L2_EVICT_BUF is None:
        _L2_EVICT_BUF = torch.empty(
            256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    return _L2_EVICT_BUF


def reference_quantize(x, rowwise, colwise):
    q = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=colwise,
    )
    return q(x)


def profile_one(name, fn, warmup):
    """Warm up `warmup` times, flush L2, then profile exactly one call.

    The L2 flush happens BEFORE torch.cuda.profiler.start() / sync, so the
    fill kernel isn't captured — NCU sees only the target kernel running
    against a cold L2."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Cold-cache: write 256 MB to evict L2, then drain so the eviction kernel
    # doesn't get profiled. By the time profiler.start() fires, GPU is idle
    # and L2 is empty.
    _l2_evict_buf().zero_()
    torch.cuda.synchronize()

    nvtx.range_push(name)
    torch.cuda.profiler.start()
    fn()
    torch.cuda.profiler.stop()
    nvtx.range_pop()
    torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--shapes", type=str, required=True,
                   help="Single shape as 'M,N'")
    p.add_argument("--direction", choices=["row", "col", "both"], required=True)
    args = p.parse_args()

    shapes = parse_shapes(args.shapes)
    if len(shapes) != 1:
        print("bench_for_ncu expects exactly one shape per invocation",
              file=sys.stderr)
        return 1

    M, N = shapes[0]
    rw, cw = {
        "row":  (True,  False),
        "col":  (False, True),
        "both": (True,  True),
    }[args.direction]

    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    # Warm the CuTeDSL JIT cache for this (rw, cw) before the profiled run
    # so JIT compilation doesn't bleed into the captured launch.
    quantize_mxfp8_cutedsl(x, rowwise=rw, colwise=cw)
    torch.cuda.synchronize()

    print(f"Profiling {M}x{N} {args.direction} (warmup={args.warmup})",
          flush=True)

    profile_one(
        f"cpp_ref_{M}x{N}_{args.direction}",
        lambda: reference_quantize(x, rw, cw),
        args.warmup,
    )
    profile_one(
        f"cutedsl_{M}x{N}_{args.direction}",
        lambda: quantize_mxfp8_cutedsl(x, rowwise=rw, colwise=cw),
        args.warmup,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
