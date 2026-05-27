"""Smoke test for _build_aot: build the .so end-to-end, load with tvm_ffi, and
run the kernel against a reference. Validates Phase 2 without C++.
"""
import os
import sys

# Mirror the test file's CUTE_DSL_ARCH pin before importing.
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
import tvm_ffi

from quantize_mxfp8_cutedsl_alt import (
    MXFP8QuantizeConfig,
    _get_aot_kernel,
    _build_aot,
    AOT_CACHE_DIR,
    SCALE_DIM,
)

M, N = 64, 64

cfg = MXFP8QuantizeConfig(
    dtype=cutlass.BFloat16, fp8_dtype="e4m3",
    rowwise=True, colwise=False,
    with_gemm_swizzled_scales=False, with_amax=False,
    activation=None, with_dbias=False, is_dact=False,
)

# Wipe any stale .so so we exercise the build path the first time.
for p in AOT_CACHE_DIR.glob("mxfp8_*"):
    p.unlink()

fn_name, so_path, active_slots, fn = _get_aot_kernel(cfg, M, N)
print(f"fn_name      = {fn_name}")
print(f"so_path      = {so_path}")
print(f"active_slots = {active_slots}")
print(f"so size      = {os.path.getsize(so_path)} bytes")
print(f"loaded fn    = {fn}")

# Runtime tensors match the AOT-time padded shape (matches TE storage).
x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
out_row = torch.empty(M, N, dtype=torch.uint8, device="cuda")
padded_M = ((M + 127) // 128) * 128
padded_S = ((N + 127) // 128) * 4
scale_row = torch.empty(padded_M, padded_S, dtype=torch.uint8, device="cuda")
noop = torch.zeros(1, dtype=torch.float32, device="cuda")     # != 1.0 → run
amax = torch.zeros(1, dtype=torch.float32, device="cuda")     # unused (WITH_AMAX=False)

# All 9 logical slots — None for inactive (mO_col, mS_col, mDbias).
fn(x, x, out_row, scale_row, None, None, noop, amax, None)
torch.cuda.synchronize()

# Reference: existing JIT path with COMPACT scale (its sym constraint requires
# mS_row.shape[0] == mX.shape[0]). The padded AOT output's first (M, N/32)
# strided sub-view should equal the compact JIT output element-wise.
from quantize_mxfp8_cutedsl_alt import _get_compiled_kernel
ref_out_row = torch.empty_like(out_row)
ref_scale_row = torch.empty(M, N // SCALE_DIM, dtype=torch.uint8, device="cuda")
jit = _get_compiled_kernel(cfg, tvm_ffi=True)
jit(x, x, ref_out_row, ref_scale_row, None, None, noop, amax, None)
torch.cuda.synchronize()

mismatch_data = (out_row != ref_out_row).sum().item()
mismatch_scale = (scale_row[:M, :N // SCALE_DIM] != ref_scale_row).sum().item()
print(f"data byte mismatches  = {mismatch_data} / {out_row.numel()}")
print(f"scale byte mismatches = {mismatch_scale} / {ref_scale_row.numel()}")
assert mismatch_data == 0 and mismatch_scale == 0, "AOT vs JIT output differs"

# Second-call cache hit (no rebuild).
fn_name2, so_path2, active_slots2, fn2 = _get_aot_kernel(cfg, M, N)
assert (fn_name, so_path, active_slots) == (fn_name2, so_path2, active_slots2)
assert fn is fn2, "expected same Function handle on cache hit"
print("cache hit OK")
