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

from cutile_kernels import (
    FusedSinkhornKnopp,
    FusedHAggregate,
    FusedHPostBDA,
    FusedProjRms,
)

from native_kernels import (
    mHCProjectionRef,
    mHCSinkhornRef,
    mHCElementwiseRef,
    mHCPreRef,
    mHCPostResRef
)

def run_sinkhorn_triton(B, T, n, dtype, device, iters, do_backward):
    nvtx.range_push("mhc_sinkhorn_triton_fwd")
    x = torch.randn((B, T, n, n), device=device, dtype=dtype, requires_grad=do_backward)
    y = mHCSinkhornOp.apply(x, n, iters)
    nvtx.range_pop()
    nvtx.range_push("mhc_sinkhorn_triton_bwd")
    if do_backward:
        y.sum().backward()
    nvtx.range_pop()

def run_sinkhorn_cutile(B, T, n, dtype, device, iters, do_backward):
    nvtx.range_push("mhc_sinkhorn_cutile_fwd")
    x = torch.randn((B, T, n, n), device=device, dtype=dtype, requires_grad=do_backward)
    y = FusedSinkhornKnopp.apply(x, iters)
    nvtx.range_pop()
    nvtx.range_push("mhc_sinkhorn_cutile_bwd")
    if do_backward:
        y.sum().backward()
    nvtx.range_pop()

def run_sinkhorn_compile(B, T, n, dtype, device, iters, do_backward):
    nvtx.range_push("mhc_sinkhorn_compile_fwd")
    x = torch.randn((B, T, n, n), device=device, dtype=dtype, requires_grad=do_backward)
    y = mHCSinkhornRef(x, n, iters)
    nvtx.range_pop()
    nvtx.range_push("mhc_sinkhorn_compile_bwd")
    if do_backward:
        y.sum().backward()
    nvtx.range_pop()

def run_sinkhorn(B, T, n, dtype, device, iters, do_backward):
    run_sinkhorn_cutile(B, T, n, dtype, device, iters, do_backward)
    run_sinkhorn_triton(B, T, n, dtype, device, iters, do_backward)
    run_sinkhorn_compile(B, T, n, dtype, device, iters, do_backward)


def run_projection_triton(B, T, n, C, dtype, device, do_backward):
    nC = n * C
    N = 2 * n + n * n
    nvtx.range_push("mhc_projection_triton_fwd")
    x = torch.randn(B * T, nC, device="cuda", requires_grad=True, dtype=dtype)
    phi = torch.randn(N, nC, dtype=dtype, requires_grad=True, device="cuda")
    Hs, r = mHCProjectionOp.apply(x, phi, n)
    nvtx.range_pop()
    nvtx.range_push("mhc_projection_triton_bwd")
    if do_backward:
        (Hs.sum() + r.sum()).backward()
    nvtx.range_pop()

def run_projection_cutile(B, T, n, C, dtype, device, do_backward):
    nC = n * C
    N = 2 * n + n * n
    nvtx.range_push("mhc_projection_cutile_fwd")
    x = torch.randn(B * T, nC, device="cuda", requires_grad=True, dtype=dtype)
    phi = torch.randn(N, nC, dtype=dtype, requires_grad=True, device="cuda")
    Hs, r = FusedProjRms.apply(x, phi)
    nvtx.range_pop()
    nvtx.range_push("mhc_projection_cutile_bwd")
    if do_backward:
        (Hs.sum() + r.sum()).backward()
    nvtx.range_pop()

def run_projection_compile(B, T, n, C, dtype, device, do_backward):
    nC = n * C
    N = 2 * n + n * n
    nvtx.range_push("mhc_projection_compile_fwd")
    x = torch.randn(B * T, nC, device="cuda", requires_grad=True, dtype=dtype)
    phi = torch.randn(N, nC, dtype=dtype, requires_grad=True, device="cuda")
    Hs, r = mHCProjectionRef(x, phi)
    nvtx.range_pop()
    nvtx.range_push("mhc_projection_compile_bwd")
    if do_backward:
        (Hs.sum() + r.sum()).backward()
    nvtx.range_pop()

def run_projection(B, T, n, C, dtype, device, do_backward):
    run_projection_cutile(B, T, n, C, dtype, device, do_backward)
    run_projection_triton(B, T, n, C, dtype, device, do_backward)
    run_projection_compile(B, T, n, C, dtype, device, do_backward)


def run_elementwise_triton(B, T, n, dtype, device, do_backward):
    N = 2 * n + n * n
    nvtx.range_push("mhc_elementwise_triton_fwd")
    H = torch.randn((B * T, 32), device=device, dtype=dtype, requires_grad=do_backward)
    alpha = torch.randn((3,), device=device, dtype=dtype, requires_grad=do_backward)
    beta = torch.randn((1, N), device=device, dtype=dtype, requires_grad=do_backward)
    r = torch.rand((B * T), device=device, dtype=dtype, requires_grad=do_backward)
    out = mHCElementwiseOp.apply(H, alpha, beta, r, n)
    nvtx.range_pop()
    nvtx.range_push("mhc_elementwise_triton_bwd")
    if do_backward:
        out.sum().backward()
    nvtx.range_pop()

def run_elementwise_compile(B, T, n, dtype, device, do_backward):
    N = 2 * n + n * n
    nvtx.range_push("mhc_elementwise_compile_fwd")
    H = torch.randn((B * T, 32), device=device, dtype=dtype, requires_grad=do_backward)
    alpha = torch.randn((3,), device=device, dtype=dtype, requires_grad=True)
    beta = torch.randn((1, N), device=device, dtype=dtype, requires_grad=True)
    r = torch.rand((B * T), device=device, dtype=dtype, requires_grad=True)
    out = mHCElementwiseRef(H[:, :N], alpha, beta, r, n)
    nvtx.range_pop()
    nvtx.range_push("mhc_elementwise_compile_bwd")
    if do_backward:
        out.sum().backward()
    nvtx.range_pop()

def run_elementwise(B, T, n, dtype, device, do_backward):
    run_elementwise_triton(B, T, n, dtype, device, do_backward)
    run_elementwise_compile(B, T, n, dtype, device, do_backward)


def run_pre_triton(B, T, n, C, dtype, device, do_backward):
    nvtx.range_push("mhc_pre_triton_fwd")
    x = torch.randn(B, T, n, C, dtype=dtype, requires_grad=True, device=device)
    H_pre = torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device)
    out = mHCPreOp.apply(x, H_pre, n)
    nvtx.range_pop()
    nvtx.range_push("mhc_pre_triton_bwd")
    if do_backward:
        out.sum().backward()
    nvtx.range_pop()

def run_pre_cutile(B, T, n, C, dtype, device, do_backward):
    nvtx.range_push("mhc_pre_cutile_fwd")
    x = torch.randn(B, T, n, C, dtype=dtype, requires_grad=True, device=device)
    H_pre = torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device)
    out = FusedHAggregate.apply(x, H_pre)
    nvtx.range_pop()
    nvtx.range_push("mhc_pre_cutile_bwd")
    if do_backward:
        out.sum().backward()
    nvtx.range_pop()

def run_pre_compile(B, T, n, C, dtype, device, do_backward):
    nvtx.range_push("mhc_pre_compile_fwd")
    x = torch.randn(B, T, n, C, dtype=dtype, requires_grad=True, device=device)
    H_pre = torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device)
    out = mHCPreRef(x, H_pre, n)
    nvtx.range_pop()
    nvtx.range_push("mhc_pre_compile_bwd")
    if do_backward:
        out.sum().backward()
    nvtx.range_pop()

def run_pre(B, T, n, C, dtype, device, do_backward):
    run_pre_cutile(B, T, n, C, dtype, device, do_backward)
    run_pre_triton(B, T, n, C, dtype, device, do_backward)
    run_pre_compile(B, T, n, C, dtype, device, do_backward)


def run_post_res_triton(B, T, n, C, dtype, device, do_backward):
    nvtx.range_push("mhc_post_res_triton_fwd")
    x = torch.randn(B, T, n, C, dtype=dtype, requires_grad=True, device=device)
    H_post = torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device)
    H_res = torch.randn(B, T, n, n, dtype=dtype, requires_grad=True, device=device)
    f = torch.randn(B, T, C, dtype=dtype, requires_grad=True, device=device)
    out = mHCPostResOp.apply(f, H_post, x, H_res, n)
    nvtx.range_pop()
    nvtx.range_push("mhc_post_res_triton_bwd")
    if do_backward:
        out.sum().backward()
    nvtx.range_pop()

def run_post_res_cutile(B, T, n, C, dtype, device, do_backward):
    nvtx.range_push("mhc_post_res_cutile_fwd")
    x = torch.randn(B, T, n, C, dtype=dtype, requires_grad=True, device=device)
    H_post = torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device)
    H_res = torch.randn(B, T, n, n, dtype=dtype, requires_grad=True, device=device)
    f = torch.randn(B, T, C, dtype=dtype, requires_grad=True, device=device)
    out = FusedHPostBDA.apply(H_res, x, H_post, f, None)
    nvtx.range_pop()
    nvtx.range_push("mhc_post_res_cutile_bwd")
    if do_backward:
        out.sum().backward()
    nvtx.range_pop()

def run_post_res_compile(B, T, n, C, dtype, device, do_backward):
    nvtx.range_push("mhc_post_res_compile_fwd")
    x = torch.randn(B, T, n, C, dtype=dtype, requires_grad=True, device=device)
    H_post = torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device)
    H_res = torch.randn(B, T, n, n, dtype=dtype, requires_grad=True, device=device)
    f = torch.randn(B, T, C, dtype=dtype, requires_grad=True, device=device)
    out = mHCPostResRef(f, H_post, x, H_res, n)
    nvtx.range_pop()
    nvtx.range_push("mhc_post_res_compile_bwd")
    if do_backward:
        out.sum().backward()
    nvtx.range_pop()

def run_post_res(B, T, n, C, dtype, device, do_backward):
    run_post_res_cutile(B, T, n, C, dtype, device, do_backward)
    run_post_res_triton(B, T, n, C, dtype, device, do_backward)
    run_post_res_compile(B, T, n, C, dtype, device, do_backward)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["sinkhorn", "projection", "elementwise", "pre", "post_res", "all"], required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
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
    dtype = torch.bfloat16

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
