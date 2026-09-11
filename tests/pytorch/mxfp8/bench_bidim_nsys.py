"""Drive the bidimensional MXFP8 cast (cast_bidim.cu) for nsys kernel timing.

Runs bf16 -> MXFP8 rowwise+colwise quantize over a list of shapes.  Nothing
here times anything: nsys measures the kernel.  The only job of this script is
to issue a clean, L2-cold stream of launches, one distinguishable grid per
shape, with the warmup launches up front so the analyzer can drop them.
"""

import argparse

import torch
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch import MXFP8Quantizer


def make_quantizer():
    q = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    q.internal = True          # raw QuantizedTensor, no Float8Tensor wrapper
    return q


def run_shape(M, N, warmup, iters, rotate):
    """Launch the bidim cast `warmup + iters` times over rotating inputs."""
    quantizer = make_quantizer()
    xs = [torch.randn(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(rotate)]

    for i in range(warmup):
        tex.quantize(xs[i % rotate], quantizer)
    torch.cuda.synchronize()

    for i in range(iters):
        tex.quantize(xs[i % rotate], quantizer)
    torch.cuda.synchronize()

    del xs
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="16384x7168;16384x8192;32768x7168;"
                                       "32768x8192;32768x16384")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rotate", type=int, default=2,
                   help="Distinct input buffers, so consecutive launches do not "
                        "hit a warm L2.")
    args = p.parse_args()

    for spec in args.shapes.split(";"):
        if not spec.strip():
            continue
        M, N = (int(v) for v in spec.strip().split("x"))
        grid = (N // 256, M // 32)   # narrow-tile grid; identifies the shape
        print(f"==> {M}x{N}  (narrow-tile grid {grid[0]}x{grid[1]})", flush=True)
        run_shape(M, N, args.warmup, args.iters, args.rotate)


if __name__ == "__main__":
    main()
