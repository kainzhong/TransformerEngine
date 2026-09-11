# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Hash quantize outputs across MXFP8 / NVFP4 / FP8-gated for one build.

Run once per arm with the same seeds; the two JSONs must be identical. A perf
change that alters results is not a perf change.

Lives in the repo (not /tmp) on purpose: `python /tmp/foo.py` puts /tmp on
sys.path, so the local build stops shadowing the prebuilt transformer_engine in
dist-packages and you silently measure the wrong library. The recorded lib md5
is the backstop.
"""

import argparse
import hashlib
import json
import os

import torch

import transformer_engine.pytorch as te  # must precede transformer_engine_torch
import transformer_engine_torch as tex
from transformer_engine.pytorch import Float8Quantizer, MXFP8Quantizer, NVFP4Quantizer


def mapped_lib():
    for line in open("/proc/self/maps"):
        if "libtransformer_engine" in line:
            return line.split()[-1]
    return "?"


def lib_md5():
    p = mapped_lib()
    if p == "?" or not os.path.exists(p):
        return "?"
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def th(t):
    return hashlib.md5(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def digest(qt, extra=None):
    parts = []
    for a in ("_rowwise_data", "_columnwise_data", "_rowwise_scale_inv", "_columnwise_scale_inv",
              "_data", "_scale_inv"):
        v = getattr(qt, a, None)
        parts.append(th(v) if v is not None else "-")
    if extra is not None:
        parts.append(th(extra))
    return "/".join(parts)


SHAPES = [(4096, 4096), (2048, 12288), (4096, 14336)]
DIRS = {"row": (True, False), "col": (False, True), "both": (True, True)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = {}

    def record(key, fn):
        try:
            r = fn()
            torch.cuda.synchronize()
        except Exception as e:  # pylint: disable=broad-except
            out[key] = f"SKIP:{type(e).__name__}"
            return
        qt = r[0] if isinstance(r, (list, tuple)) else r
        extra = r[1] if isinstance(r, (list, tuple)) and len(r) > 1 and torch.is_tensor(r[1]) else None
        out[key] = digest(qt, extra)

    for M, N in SHAPES:
        for d, (row, col) in DIRS.items():
            torch.manual_seed(1234)
            x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
            xg = torch.randn(M, 2 * N, dtype=torch.bfloat16, device="cuda")

            # --- MXFP8: plain / act / dact / dbias -------------------------
            def mxq():
                q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=row, columnwise=col)
                q.internal = True
                return q

            record(f"mxfp8_plain|{d}|{M}x{N}", lambda: tex.quantize(x, mxq()))
            record(f"mxfp8_gelu|{d}|{M}x{N}", lambda: tex.gelu(x, mxq()))
            record(f"mxfp8_dgelu|{d}|{M}x{N}", lambda: tex.dgelu(x, x, mxq()))
            record(f"mxfp8_dbias|{d}|{M}x{N}", lambda: tex.bgrad_quantize(x, mxq()))
            # --- MXFP8 gated ----------------------------------------------
            record(f"mxfp8_swiglu|{d}|{M}x{N}", lambda: tex.swiglu(xg, mxq()))
            record(f"mxfp8_geglu|{d}|{M}x{N}", lambda: tex.geglu(xg, mxq()))

            # --- NVFP4 1D / 2D ---------------------------------------------
            for mode in ("1d", "2d"):
                def nvq(mode=mode):
                    q = NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1, rowwise=row,
                                       columnwise=col, with_2d_quantization=(mode == "2d"))
                    q.internal = True
                    return q

                record(f"nvfp4{mode}|{d}|{M}x{N}", lambda mode=mode: tex.quantize(x, nvq(mode)))

            # --- FP8 gated (the newly found site) --------------------------
            def f8q():
                scale = torch.ones(1, dtype=torch.float32, device="cuda")
                amax = torch.zeros(1, dtype=torch.float32, device="cuda")
                return Float8Quantizer(scale=scale, amax=amax, fp8_dtype=tex.DType.kFloat8E4M3)

            record(f"fp8_swiglu|{d}|{M}x{N}", lambda: tex.swiglu(xg, f8q()))
            record(f"fp8_geglu|{d}|{M}x{N}", lambda: tex.geglu(xg, f8q()))

            del x, xg
            torch.cuda.empty_cache()

    md5 = lib_md5()
    json.dump({"lib": mapped_lib(), "lib_md5": md5, "hashes": out}, open(args.out, "w"), indent=1)
    ok = sum(1 for v in out.values() if not str(v).startswith("SKIP"))
    print(f"lib {md5[:12]}  entries {len(out)} ({ok} ran, {len(out)-ok} skipped) -> {args.out}")


if __name__ == "__main__":
    main()
