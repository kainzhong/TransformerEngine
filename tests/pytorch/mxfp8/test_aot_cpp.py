"""End-to-end test: AOT-dispatched quantize via the C++ entry, compared
against the existing JIT path for bit-identity."""
import os
import sys

def _detect_arch():
    try:
        import torch
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        return f"sm_{major}{minor}"
    except Exception:
        return "sm_100a"

os.environ.setdefault("CUTE_DSL_ARCH", _detect_arch())
sys.path.insert(0, os.path.dirname(__file__))

import torch
import cutlass
import transformer_engine.pytorch  # loads libtransformer_engine.so
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

from quantize_mxfp8_cutedsl_alt import (
    MXFP8QuantizeConfig, quantize_mxfp8_cutedsl_aot,
    _get_compiled_kernel, AOT_CACHE_DIR, SCALE_DIM,
)

M, N = 64, 64
torch.manual_seed(0)
x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

# Force a clean build for the test.
for p in AOT_CACHE_DIR.glob("mxfp8_*"):
    p.unlink()

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
quantizer.internal = True

# Single-call entry: builds the .so on cache miss, then dispatches via C++.
out = quantize_mxfp8_cutedsl_aot(x, quantizer)
torch.cuda.synchronize()
print(f"out type     = {type(out).__name__}")
print(f"rowwise_data shape       = {tuple(out._rowwise_data.shape)}")
print(f"rowwise_scale_inv shape  = {tuple(out._rowwise_scale_inv.shape)}")

# Reference: JIT path with compact-shape scale tensor. AOT-time and JIT
# write the same VALUES at logical [i, j], but the AOT layout is padded
# (stride differs). Compare the compact prefix.
cfg = MXFP8QuantizeConfig(
    dtype=cutlass.BFloat16, fp8_dtype="e4m3",
    rowwise=True, colwise=False,
    with_gemm_swizzled_scales=False, with_amax=False,
    activation=None, with_dbias=False, is_dact=False)
ref_out_row = torch.empty(M, N, dtype=torch.uint8, device="cuda")
ref_scale_row = torch.empty(M, N // SCALE_DIM, dtype=torch.uint8, device="cuda")
noop = torch.zeros(1, dtype=torch.float32, device="cuda")
amax = torch.zeros(1, dtype=torch.float32, device="cuda")
jit = _get_compiled_kernel(cfg, tvm_ffi=True)
jit(x, x, ref_out_row, ref_scale_row, None, None, noop, amax, None)
torch.cuda.synchronize()

aot_scale_view = out._rowwise_scale_inv[:M, :N // SCALE_DIM]
mismatch_data = (out._rowwise_data != ref_out_row).sum().item()
mismatch_scale = (aot_scale_view != ref_scale_row).sum().item()
print(f"data byte mismatches  = {mismatch_data} / {out._rowwise_data.numel()}")
print(f"scale byte mismatches = {mismatch_scale} / {aot_scale_view.numel()}")
assert mismatch_data == 0 and mismatch_scale == 0
print("C++ AOT dispatch == JIT bit-identical OK")

# Second call hits both Python and C++ caches.
out2 = quantize_mxfp8_cutedsl_aot(x, quantizer)
torch.cuda.synchronize()
mismatch2 = (out2._rowwise_data != out._rowwise_data).sum().item()
assert mismatch2 == 0
print("cache hit OK")
