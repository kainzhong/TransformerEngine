import argparse
import os
import time

import torch
from torch.cuda import nvtx
import torch.cuda.profiler as profiler

from transformer_engine.pytorch.triton.mhc import (
    mHCElementwiseOp,
    mHCPostResOp,
    mHCPreOp,
    mHCProjectionOp,
    mHCSinkhornOp,
)

def run_sinkhorn(B, T, n, dtype, device, iters, do_backward):
    x = torch.randn((B, T, n, n), device=device, dtype=dtype, requires_grad=do_backward)
    y = mHCSinkhornOp.apply(x, n, iters)
    if do_backward:
        y.sum().backward()
    return y


def run_projection(B, T, n, C, dtype, device, do_backward):
    nC = n * C
    N = 2 * n + n * n
    x = torch.randn(B * T, nC, device="cuda", requires_grad=True, dtype=dtype)
    phi = torch.randn(N, nC, dtype=dtype, requires_grad=True, device="cuda")
    Hs, r = mHCProjectionOp.apply(x, phi, n)
    if do_backward:
        (Hs.sum() + r.sum()).backward()
    return Hs, r


def run_elementwise(B, T, n, dtype, device, do_backward):
    N = 2 * n + n * n
    H = torch.randn((B * T, 32), device=device, dtype=dtype, requires_grad=do_backward)
    alpha = torch.randn((3,), device=device, dtype=dtype, requires_grad=do_backward)
    beta = torch.randn((1, N), device=device, dtype=dtype, requires_grad=do_backward)
    r = torch.rand((B * T), device=device, dtype=dtype, requires_grad=do_backward)
    out = mHCElementwiseOp.apply(H, alpha, beta, r, n)
    if do_backward:
        out.sum().backward()
    return out

def run_pre(B, T, n, C, dtype, device, do_backward):
    x = torch.randn(B, T, n, C, dtype=dtype, requires_grad=True, device=device)
    H_pre = torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device)
    out = mHCPreOp.apply(x, H_pre, n)
    if do_backward:
        out.sum().backward()
    return out

def run_post_res(B, T, n, C, dtype, device, do_backward):
    x = torch.randn(B, T, n, C, dtype=dtype, requires_grad=True, device=device)
    H_post = torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device)
    H_res = torch.randn(B, T, n, n, dtype=dtype, requires_grad=True, device=device)
    f = torch.randn(B, T, C, dtype=dtype, requires_grad=True, device=device)
    out = mHCPostResOp.apply(f, H_post, x, H_res, n)
    if do_backward:
        out.sum().backward()
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["sinkhorn", "projection", "elementwise", "pre", "post_res", "all"], required=True)
    parser.add_argument("--dtype", choices=["float32"], default="float32")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument("--B", type=int, default=32)
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--C", type=int, default=4096)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float32

    do_backward = not args.no_backward

    B = args.B
    T = args.T
    C = args.C
    n = 4

    print(f"Running {args.operation} with B={B}, T={T}")

    # Warmup
    for _ in range(args.warmup):
        if args.operation == "sinkhorn":
            run_sinkhorn(B, T, n, dtype, device, args.sinkhorn_iters, do_backward)
        elif args.operation == "projection":
            run_projection(B, T, n, C, dtype, device, do_backward)
        elif args.operation == "elementwise":
            run_elementwise(B, T, n, dtype, device, do_backward)
        elif args.operation == "pre":
            run_pre(B, T, n, C, dtype, device, do_backward)
        elif args.operation == "post_res":
            run_post_res(B, T, n, C, dtype, device, do_backward)
        elif args.operation == "all":
            run_sinkhorn(B, T, n, dtype, device, args.sinkhorn_iters, do_backward)
            run_projection(B, T, n, C, dtype, device, do_backward)
            run_elementwise(B, T, n, dtype, device, do_backward)
            run_pre(B, T, n, C, dtype, device, do_backward)
            run_post_res(B, T, n, C, dtype, device, do_backward)
    torch.cuda.synchronize()

    # Start profiling AFTER warmup/autotuning
    torch.cuda.cudart().cudaProfilerStart()

    nvtx_label = f"{args.operation}_B{B}_T{T}_C{C}_bw{int(do_backward)}"

    # Profile iterations
    nvtx.range_push(nvtx_label)
    for _ in range(args.iters):
        if args.operation == "sinkhorn":
            run_sinkhorn(B, T, n, dtype, device, args.sinkhorn_iters, do_backward)
        elif args.operation == "projection":
            run_projection(B, T, n, C, dtype, device, do_backward)
        elif args.operation == "elementwise":
            run_elementwise(B, T, n, dtype, device, do_backward)
        elif args.operation == "pre":
            run_pre(B, T, n, C, dtype, device, do_backward)
        elif args.operation == "post_res":
            run_post_res(B, T, n, C, dtype, device, do_backward)
        elif args.operation == "all":
            run_sinkhorn(B, T, n, dtype, device, args.sinkhorn_iters, do_backward)
            run_projection(B, T, n, C, dtype, device, do_backward)
            run_elementwise(B, T, n, dtype, device, do_backward)
            run_pre(B, T, n, C, dtype, device, do_backward)
            run_post_res(B, T, n, C, dtype, device, do_backward)
    torch.cuda.synchronize()
    nvtx.range_pop()

    # Stop profiling
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
