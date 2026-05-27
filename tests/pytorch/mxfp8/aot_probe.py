"""Phase-1 probe: AOT-compile one MXFP8 cfg with the CuTe ABI (no TVM FFI, no
mark_layout_dynamic) and dump the generated header so we can lock the C++ ABI.

Two cfgs are exported:
  - "full"    : all 9 tensor args are real (rowwise + colwise + amax + dbias).
  - "minimal" : 6 None args (rowwise-only, no amax, no dbias) so we can see
                whether None slots are dropped from the wrapper or kept as
                null-tensor placeholders.

Run:
    python3 tests/pytorch/mxfp8/aot_probe.py

After running, paste back the contents of `aot_probe_out/*.h`.
"""
import os
import sys

# Pin to current device's SM (mirrors quantize_mxfp8_cutedsl_alt.py).
def _detect_arch() -> str:
    try:
        import torch
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        return f"sm_{major}{minor}"
    except Exception:
        return "sm_100a"

os.environ.setdefault("CUTE_DSL_ARCH", _detect_arch())

# Reuse the existing kernel definition verbatim.
sys.path.insert(0, os.path.dirname(__file__))
from quantize_mxfp8_cutedsl_alt import MXFP8QuantizeSmemKernel, MXFP8QuantizeConfig

import cutlass
import cutlass.cute as cute
from cutlass import Float32

OUT_DIR = os.path.join(os.path.dirname(__file__), "aot_probe_out")
os.makedirs(OUT_DIR, exist_ok=True)

# Small literal shape; we want constexpr shapes to see if shape arrays vanish.
M, N = 64, 64
SCALE_DIM = 32

# Row-major, gmem, aligned. Same as _get_compiled_kernel but with LITERAL int
# dims rather than sym_int — that's the whole point of Phase 1 (does the
# generated struct still carry `dynamic_shapes[]` or does it collapse?).
kw_rm16 = dict(stride_order=(1, 0),
               memspace=cute.AddressSpace.gmem, assumed_align=16)
kw_rm4 = dict(stride_order=(1, 0),
              memspace=cute.AddressSpace.gmem, assumed_align=4)


def make_fake(dtype, shape, align16=True):
    base = kw_rm16 if align16 else kw_rm4
    if len(shape) == 1:
        # 1D tensors take stride_order=(0,) not (1,0).
        kw = {**base, "stride_order": (0,)}
    else:
        kw = base
    return cute.runtime.make_fake_compact_tensor(dtype, shape, **kw)


def export_cfg(label, cfg, args, tvm_ffi=False):
    print(f"\n==== {label} cfg (tvm_ffi={tvm_ffi}) ====")
    print(f"  ROWWISE={cfg.ROWWISE} COLWISE={cfg.COLWISE} "
          f"WITH_AMAX={cfg.WITH_AMAX} WITH_DBIAS={cfg.WITH_DBIAS} "
          f"IS_DACT={cfg.IS_DACT}")
    print(f"  non-None args: {[i for i, a in enumerate(args) if a is not None]}")
    kernel_obj = MXFP8QuantizeSmemKernel(cfg)
    fn_name = f"mxfp8_probe_{label}"
    if tvm_ffi:
        # TVM FFI export — predictable symbol __tvm_ffi_<function_name>.
        compiled = cute.compile(kernel_obj, *args, options="--enable-tvm-ffi")
        o_path = os.path.join(OUT_DIR, f"{fn_name}.o")
        compiled.export_to_c(o_path, function_name=fn_name)
    else:
        compiled = cute.compile(kernel_obj, *args)
        compiled.export_to_c(
            file_path=OUT_DIR, file_name=fn_name, function_prefix=fn_name,
        )
        o_path = os.path.join(OUT_DIR, f"{fn_name}.o")
    print(f"  -> {o_path} ({os.path.getsize(o_path)} bytes)")
    return o_path


# ---- cfg 1: "all_active" — rowwise only, no extras (kept as a sibling for
# comparison vs minimal; same active arg count as minimal here). Once we have
# the base ABI we can add amax/dbias/dact in a follow-up probe.
cfg_full = MXFP8QuantizeConfig(
    dtype=cutlass.BFloat16, fp8_dtype="e4m3",
    rowwise=True, colwise=False,
    with_gemm_swizzled_scales=False, with_amax=False,
    activation=None, with_dbias=False, is_dact=False,
)
in_full          = make_fake(cutlass.BFloat16, (M, N))
out_row_full     = make_fake(cute.Uint8, (M, N))
scale_row_full   = make_fake(cute.Uint8, (M, N // SCALE_DIM))
noop_full        = make_fake(Float32, (1,), align16=False)
amax_full        = make_fake(Float32, (1,), align16=False)

args_full = (
    in_full,          # mX
    in_full,          # mActIn (aliased)
    out_row_full,     # mO_row
    scale_row_full,   # mS_row
    None,             # mO_col
    None,             # mS_col
    noop_full,        # mNoop
    amax_full,        # mAmax
    None,             # mDbias
)

h_full = export_cfg("all_active", cfg_full, args_full)
o_full_ffi = export_cfg("all_active_ffi", cfg_full, args_full, tvm_ffi=True)

# ---- cfg 2: "minimal" — rowwise-only, no amax, no dbias, no dact ----
# Drops 3 args to None: mO_col, mS_col, mDbias.
# (mNoop and mAmax stay real because the kernel signature has them as
# `cute.Tensor`, not Optional, in the current code.)
cfg_min = MXFP8QuantizeConfig(
    dtype=cutlass.BFloat16, fp8_dtype="e4m3",
    rowwise=True, colwise=False,
    with_gemm_swizzled_scales=False, with_amax=False,
    activation=None, with_dbias=False, is_dact=False,
)
in_min          = make_fake(cutlass.BFloat16, (M, N))
out_row_min     = make_fake(cute.Uint8, (M, N))
scale_row_min   = make_fake(cute.Uint8, (M, N // SCALE_DIM))
noop_min        = make_fake(Float32, (1,), align16=False)
amax_min        = make_fake(Float32, (1,), align16=False)

args_min = (
    in_min,          # mX
    in_min,          # mActIn
    out_row_min,     # mO_row
    scale_row_min,   # mS_row
    None,            # mO_col
    None,            # mS_col
    noop_min,        # mNoop
    amax_min,        # mAmax
    None,            # mDbias
)

h_min = export_cfg("minimal", cfg_min, args_min)

print("\nGenerated headers — paste these back so we can read the ABI:")
for p in (h_full, h_min):
    print(f"  {p}")
