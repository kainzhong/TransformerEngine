# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Render the five-fix A/B (cold and warm L2) as markdown.

Everything is pure GPU kernel time: the median CUPTI kernel duration from nsys
(`nvtx_kern_sum`), counting only the quantize kernel under test. No host time --
only kernel bodies changed, host dispatch overlaps with GPU execution in a real
step, and it disappears entirely under CUDA graphs.
"""

import json
import os
import statistics
from collections import defaultdict

from nsys_load import load

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "PROVENANCE_FIX_RESULTS.md")

SHAPE_NOTE = {
    "4096x4096": "Llama-3-8B hidden", "4096x14336": "Llama-3-8B FFN",
    "8192x8192": "Llama-3-70B hidden", "8192x28672": "Llama-3-70B FFN",
    "2048x12288": "GPT-3-175B hidden", "16384x5120": "long-seq hidden",
}

FIXES = [
    ("generic", "`cast/mxfp8/quantize_mxfp8.cuh`", "MXFP8 generic quantize kernel"),
    ("specialized", "`cast/mxfp8/specialized/quantize_mxfp8.cuh`",
     "MXFP8 specialized cast-only kernel (bidimensional)"),
    ("mxfp8_gated", "`cast/mxfp8/gated_mxfp8.cuh`", "MXFP8 gated (swiglu / geglu) kernel"),
    ("nvfp4", "`cast/nvfp4/quantize_transpose_nvfp4.cuh`", "NVFP4 2D quantize-transpose kernel"),
    ("fp8_gated", "`cast/fp8/gated_fp8.cuh`", "FP8 gated (swiglu / geglu) kernel"),
]

# label prefix -> (kernel_match, forced family or None to classify by kernel name)
SOURCES = [("f2", "mxfp8", None), ("f2nv", "nvfp4", "nvfp4"), ("f2f8", "fp8_gated", "fp8_gated")]


def classify(kern):
    if "cast_only" in kern:
        return "specialized"
    if "gated" in kern:
        return "mxfp8_gated"
    return "generic"


def collect():
    rows = {}
    for pfx, km, forced in SOURCES:
        data = {}
        for arm in ("base", "helper"):
            for reg in ("cold", "warm"):
                data[(arm, reg)] = load(f"{pfx}_{arm}_{reg}", kernel_match=km)[0]
        for k, v in data[("base", "cold")].items():
            try:
                rows[f"{pfx}:{k}"] = {
                    "fix": forced or classify(v["kernel"]),
                    "combo": v["combo"], "dir": v["dir"], "M": v["M"], "N": v["N"],
                    "cold_b": v["us"], "cold_a": data[("helper", "cold")][k]["us"],
                    "warm_b": data[("base", "warm")][k]["us"],
                    "warm_a": data[("helper", "warm")][k]["us"],
                }
            except KeyError:
                continue
    return rows


def prov():
    out = {}
    for arm in ("base", "helper"):
        out[arm] = json.load(open(os.path.join(HERE, "nsys", f"f2_{arm}_cold_manifest.json")))["provenance"]
    return out


def main():
    rows = collect()
    pv = prov()
    L = []
    A = L.append

    A("# MXFP8 / NVFP4 / FP8 shared-memory provenance fix -- results")
    A("")
    A(
        "Aligning the dynamic-SHMEM base through an integer and casting the *integer* back to a "
        "pointer loses the link to the `extern __shared__` object. ptxas can then no longer prove "
        "the address is in the shared window, and falls back to generic address-space accesses "
        "(`LD.E`/`ST.E`) that resolve the window at runtime. Deriving the aligned address from the "
        "original pointer by pointer arithmetic restores `LDS`/`STS`; the shared `align_up()` "
        "helper in `common/utils.cuh` is the single place that idiom now lives."
    )
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| before | `{pv['base']['mapped_lib_md5'][:12]}` (main, no fixes) |")
    A(f"| after | `{pv['helper']['mapped_lib_md5'][:12]}` (all five fixes) |")
    A(f"| GPU | {pv['base']['gpu']} |")
    A(f"| host | `{pv['base'].get('host','?')}` (both arms; kernel times only compare within a node) |")
    A("| timing | pure GPU kernel time, median CUPTI duration via nsys `nvtx_kern_sum` |")
    A("")

    A("## Summary")
    A("")
    A("| fix | file | n | cold median | cold best | warm median | warm best |")
    A("|---|---|--:|--:|--:|--:|--:|")
    for key, path, _ in FIXES:
        sel = [r for r in rows.values() if r["fix"] == key]
        if not sel:
            continue
        c = [r["cold_b"] / r["cold_a"] for r in sel]
        w = [r["warm_b"] / r["warm_a"] for r in sel]
        A(f"| {key} | {path} | {len(sel)} | **{statistics.median(c):.3f}x** | {max(c):.3f}x | "
          f"**{statistics.median(w):.3f}x** | {max(w):.3f}x |")
    allr = list(rows.values())
    c = [r["cold_b"] / r["cold_a"] for r in allr]
    w = [r["warm_b"] / r["warm_a"] for r in allr]
    A(f"| **all** | | **{len(allr)}** | **{statistics.median(c):.3f}x** | {max(c):.3f}x | "
      f"**{statistics.median(w):.3f}x** | {max(w):.3f}x |")
    A("")
    tb, ta = sum(r["cold_b"] for r in allr), sum(r["cold_a"] for r in allr)
    wb, wa = sum(r["warm_b"] for r in allr), sum(r["warm_a"] for r in allr)
    A(f"Total kernel time over all {len(allr)} workloads: "
      f"**cold {tb:.0f} -> {ta:.0f} us ({tb/ta:.3f}x)**, "
      f"**warm {wb:.0f} -> {wa:.0f} us ({wb/wa:.3f}x)**. "
      f"Worst single workload: cold **{min(c):.3f}x**, warm **{min(w):.3f}x**.")
    A("")

    for key, path, title in FIXES:
        sel = [r for r in rows.values() if r["fix"] == key]
        if not sel:
            continue
        A(f"## {title}")
        A("")
        A(f"{path}")
        A("")
        A("| fusion | dir | n | cold median | cold range | warm median | warm range |")
        A("|---|---|--:|--:|--:|--:|--:|")
        fam = defaultdict(list)
        for r in sel:
            fam[(r["combo"], r["dir"])].append(r)
        for (combo, d), v in sorted(
            fam.items(), key=lambda kv: -statistics.median([x["cold_b"]/x["cold_a"] for x in kv[1]])
        ):
            cc = [x["cold_b"]/x["cold_a"] for x in v]
            ww = [x["warm_b"]/x["warm_a"] for x in v]
            A(f"| `{combo}` | {d} | {len(v)} | **{statistics.median(cc):.3f}x** | "
              f"{min(cc):.3f}-{max(cc):.3f} | **{statistics.median(ww):.3f}x** | "
              f"{min(ww):.3f}-{max(ww):.3f} |")
        A("")

    A("<details>")
    A("<summary>Full per-workload numbers (us, pure GPU kernel time)</summary>")
    A("")
    A("| fix | fusion | dir | shape | model tensor | cold before | cold after | cold | "
      "warm before | warm after | warm |")
    A("|---|---|---|---|---|--:|--:|--:|--:|--:|--:|")
    for k, r in sorted(rows.items(), key=lambda kv: (kv[1]["fix"], kv[1]["combo"], kv[1]["dir"],
                                                     -(kv[1]["cold_b"]/kv[1]["cold_a"]))):
        shape = f"{r['M']}x{r['N']}"
        A(f"| {r['fix']} | `{r['combo']}` | {r['dir']} | {shape} | {SHAPE_NOTE.get(shape,'')} | "
          f"{r['cold_b']:.1f} | {r['cold_a']:.1f} | {r['cold_b']/r['cold_a']:.3f}x | "
          f"{r['warm_b']:.1f} | {r['warm_a']:.1f} | {r['warm_b']/r['warm_a']:.3f}x |")
    A("")
    A("</details>")
    A("")

    A("## Patched but not measurable")
    A("")
    A("Three further sites carry the identical anti-pattern but cannot be reached, so they are "
      "fixed for consistency rather than measured -- leaving them would let the pattern be "
      "reintroduced if those paths are ever enabled.")
    A("")
    A("| site | why unreachable |")
    A("|---|---|")
    A("| `cast/mxfp8/specialized/quantize_mxfp8.cuh` (warp-specialized variant) | guarded by "
      "`enable_if_t<CastTraits::_use_warp_specialization>`, and that flag is `false`; the library "
      "contains only the 8 non-warp-specialized instantiations |")
    A("| `cast/nvfp4/quantize_transpose_nvfp4.cuh` (1D kernel) | `quantize_transpose<false>` is "
      "only called for bf16, and its first statement diverts every bf16 1D call to "
      "`quantize_transpose_tuned_1D` |")
    A("| `cast/nvfp4/group_quantize_transpose_nvfp4.cuh` | the PyTorch grouped path requires RHT "
      "(`\"graph safe grouped quant kernel for non-RHT path is not ready yet\"`), which routes to "
      "the hadamard fusion kernels instead |")
    A("")
    A("`hadamard_transform/*.cu` carries the same pattern at 3 more sites but was deliberately "
      "left alone: across all 108 hadamard kernels there are only 72 generic `LD`/`ST` in total "
      "(worst kernel: 6), so there is nothing measurable to win.")
    A("")

    A("## Method")
    A("")
    A("- Both arms are full builds of the same tree, profiled back to back **on one node**. Each "
      "profile records the hostname and the md5 of the `libtransformer_engine.so` actually mapped "
      "into the process (read from `/proc/self/maps` after load), so neither a stale build nor a "
      "cross-node comparison can slip through unnoticed.")
    A("- **cold L2**: 512 MB scratch zeroed before every iteration. **warm L2**: no flush, "
      "back-to-back steady state. Both are pure kernel time; the cache state only changes what "
      "the kernel is bottlenecked on.")
    A("- Numerics are **bitwise identical** across the two builds (90 configurations spanning "
      "MXFP8 plain/gelu/dgelu/dbias/swiglu/geglu, NVFP4 1D and 2D, and FP8 swiglu/geglu). "
      "This is a pure codegen change.")
    A("- Gains track how scalar the shared-memory access pattern is, because the generic-address "
      "penalty is charged per instruction: paths moving ~1.5-1.7 bytes/instruction gain most, "
      "while already-vectorized paths (~10 bytes/instruction) barely move.")
    A("- Reproduce with `./run_final.sh && python gen_final_report.py`.")

    open(OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L[: L.index("<details>")]))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
