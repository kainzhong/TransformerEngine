import gc
import time

import torch
import torch.cuda.nvtx as nvtx

import cutlass.cute as cute

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

from quantize_mxfp8_cutedsl_alt import quantize_mxfp8_cutedsl

x = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,
)
quantizer.internal = True
output = tex.quantize(x, quantizer)

class DoNothingKernel:
    def __init__(self):
        pass

    @cute.jit
    def __call__(self):
        pass

compiled = cute.compile(DoNothingKernel(), options="--enable-tvm-ffi")


def warmup_cutedsl(x, warmup):
    for _ in range(warmup):
        out = tex.quantize_with_func(x, quantizer, None, None, None)
        quantize_mxfp8_cutedsl(
            x=x,
            quantized_output=out,
            rowwise=quantizer.rowwise_usage,
            colwise=quantizer.columnwise_usage,
            fp8_dtype="e4m3" if quantizer.dtype == tex.DType.kFloat8E4M3 else "e5m2",
            with_gemm_swizzled_scales=quantizer.optimize_for_gemm,
            with_amax=False,
            activation=None,
            act_input=None,
            compute_dbias=False,
            is_dact=False,
        )
    torch.cuda.synchronize()

def measure_cutedsl(x, iters, batches):
    per_iter_us = []
    for _ in range(batches):
        t0 = time.perf_counter_ns()
        for i in range(iters):
            out = tex.quantize_with_func(x, quantizer, None, None, None)
            quantize_mxfp8_cutedsl(
                x=x,
                quantized_output=out,
                rowwise=quantizer.rowwise_usage,
                colwise=quantizer.columnwise_usage,
                fp8_dtype="e4m3" if quantizer.dtype == tex.DType.kFloat8E4M3 else "e5m2",
                with_gemm_swizzled_scales=quantizer.optimize_for_gemm,
                with_amax=False,
                activation=None,
                act_input=None,
                compute_dbias=False,
                is_dact=False,
            )
        t1 = time.perf_counter_ns()
        torch.cuda.synchronize()
        per_iter_us.append((t1 - t0) / 1e3 / iters)
    avg_us = sum(per_iter_us) / len(per_iter_us)
    print(f"[CuTeDSL] Time: {avg_us:.3f} us")
    return avg_us

def measure_cutedsl_do_nothing(x, iters, batches):
    per_iter_us = []
    for _ in range(batches):
        t0 = time.perf_counter_ns()
        for i in range(iters):
            out = tex.quantize_with_func(x, quantizer, None, None, None)
            quantize_mxfp8_cutedsl(
                x=x,
                quantized_output=out,
                rowwise=quantizer.rowwise_usage,
                colwise=quantizer.columnwise_usage,
                fp8_dtype="e4m3" if quantizer.dtype == tex.DType.kFloat8E4M3 else "e5m2",
                with_gemm_swizzled_scales=quantizer.optimize_for_gemm,
                with_amax=False,
                activation=None,
                act_input=None,
                compute_dbias=False,
                is_dact=False,
                do_nothing=True
            )
        t1 = time.perf_counter_ns()
        torch.cuda.synchronize()
        per_iter_us.append((t1 - t0) / 1e3 / iters)
    avg_us = sum(per_iter_us) / len(per_iter_us)
    print(f"[CuTeDSL] Time: {avg_us:.3f} us")
    return avg_us

def warmup_cpp(x, warmup):
    for i in range(warmup):
        tex.quantize(x, quantizer)
    torch.cuda.synchronize()

def measure_cpp(x, iters, batches):
    per_iter_us = []
    for _ in range(batches):
        t0 = time.perf_counter_ns()
        for i in range(iters):
            tex.quantize(x, quantizer)
        t1 = time.perf_counter_ns()
        torch.cuda.synchronize()
        per_iter_us.append((t1 - t0) / 1e3 / iters)
    avg_us = sum(per_iter_us) / len(per_iter_us)
    print(f"[C++] Time: {avg_us:.3f} us")
    return avg_us

def warmup_baseline(x, warmup):
    for i in range(warmup):
        tex.quantize_with_func(x, quantizer, None, None, None)
    torch.cuda.synchronize()

def measure_baseline(x, iters, batches):
    per_iter_us = []
    for _ in range(batches):
        t0 = time.perf_counter_ns()
        for i in range(iters):
            tex.quantize_with_func(x, quantizer, None, None, None)
        t1 = time.perf_counter_ns()
        torch.cuda.synchronize()
        per_iter_us.append((t1 - t0) / 1e3 / iters)
    avg_us = sum(per_iter_us) / len(per_iter_us)
    print(f"[Baseline] Time: {avg_us:.3f} us")
    return avg_us


if __name__ == "__main__":
    gc.collect()
    gc.disable()
    warmup = 500
    iters = 16
    batches = 100

    warmup_baseline(x, warmup)
    warmup_cpp(x, warmup)
    warmup_cutedsl(x, warmup)

    measure_baseline(x, iters, batches)
    measure_cpp(x, iters, batches)
    measure_cutedsl(x, iters, batches)
    measure_cutedsl_do_nothing(x, iters, batches)

