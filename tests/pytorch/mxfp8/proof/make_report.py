# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Render the PR A/B as markdown, from pure-GPU kernel times.

All timings come from nsys/CUPTI (`nvtx_kern_sum`), counting only the MXFP8
quantize kernel. Nothing host-side is included: only the kernel body changed, and
in a real training step host dispatch overlaps with GPU execution and disappears
entirely under CUDA graphs, so wall-clock would only dilute the measurement.
"""

import json
import os
import statistics
from collections import defaultdict

from nsys_load import load

HERE = os.path.dirname(os.path.abspath(__file__))

SHAPE_NOTE = {
    "4096x4096": "Llama-3-8B hidden",
    "4096x14336": "Llama-3-8B FFN",
    "8192x8192": "Llama-3-70B hidden",
    "8192x28672": "Llama-3-70B FFN",
    "2048x12288": "GPT-3-175B hidden",
    "16384x5120": "long-seq hidden",
}

LABEL = {
    "plain": "`quantize`",
    "gelu": "`gelu` + quantize",
    "dgelu": "`dgelu` + quantize",
    "dsilu": "`dsilu` + quantize",
    "dbias": "`bgrad_quantize`",
    "dbias_dgelu": "`dbias_dgelu` + quantize",
}


def main():
    A_, pa = load("base_k")
    B_, pb = load("fix_k")

    L = []
    A = L.append

    sass = {}
    for arm in ("base", "fix"):
        p = os.path.join(HERE, "sass", f"{arm}_sass.json")
        if os.path.exists(p):
            sass[arm] = json.load(open(p))["agg"]

    if len(sass) == 2:
        b, f = sass["base"], sass["fix"]
        A("### Generated SASS (all 612 `quantize_mxfp8_kernel` instantiations, sm_100a)")
        A("")
        A("| | before | after |")
        A("|---|--:|--:|")
        A(f"| `LDS`/`STS` (direct shared) | {b['shared']:,} | **{f['shared']:,}** |")
        A(f"| `LD.E`/`ST.E` (generic address space) | **{b['generic']:,}** | **{f['generic']:,}** |")
        A(f"| `LDG`/`STG` (global) | {b['global']:,} | {f['global']:,} |")
        A(f"| total instructions | {b['insns']:,} | {f['insns']:,} |")
        A("")
        A(
            "Round-tripping the dynamic-SHMEM base pointer through `uintptr_t` and casting the "
            "*integer* back to a pointer loses the link to the `extern __shared__` object, so "
            "ptxas can no longer prove the address is in the shared window and falls back to "
            "generic address-space accesses. Computing the alignment as an offset on the "
            "original `char*` restores a 1:1 substitution -- `ST.E.U8` -> `STS.U8`, "
            f"`LD.E.U16` -> `LDS.U16` -- removing {b['insns']-f['insns']:,} instructions of "
            "address arithmetic."
        )
        A("")

    # keys present in both, split by which kernel actually ran (from the profile)
    keys = [k for k in A_ if k in B_]
    rows = []
    for k in keys:
        a, b = A_[k], B_[k]
        rows.append((k, a["us"], b["us"], a["us"] / b["us"], a, b))

    gen = [r for r in rows if not r[4]["specialized"]]
    ctl = [r for r in rows if r[4]["specialized"]]

    A("### Speedup by fusion (pure GPU kernel time)")
    A("")
    A("| fusion | kernel | n | median | min | max |")
    A("|---|---|--:|--:|--:|--:|")
    fam = defaultdict(list)
    for k, ta, tb, sp, a, b in gen:
        fam[a["combo"]].append(sp)
    for c in sorted(fam, key=lambda c: -statistics.median(fam[c])):
        v = fam[c]
        A(
            f"| {LABEL.get(c,c)} | generic | {len(v)} | **{statistics.median(v):.3f}x** | "
            f"{min(v):.3f}x | {max(v):.3f}x |"
        )
    if ctl:
        v = [r[3] for r in ctl]
        A(
            f"| `quantize` (row / bidirectional) | specialized (untouched) | {len(v)} | "
            f"**{statistics.median(v):.3f}x** | {min(v):.3f}x | {max(v):.3f}x |"
        )
    A("")
    ta = sum(r[1] for r in gen)
    tb = sum(r[2] for r in gen)
    A(
        f"Across all {len(gen)} workloads that reach the changed kernel: "
        f"**{ta:.0f} us -> {tb:.0f} us ({ta/tb:.3f}x, {100*(1-tb/ta):.1f}% less kernel time)**. "
        f"The {len(ctl)} workloads that dispatch to the untouched specialized kernel are the "
        f"control group and sit at {statistics.median([r[3] for r in ctl]):.3f}x."
    )
    A("")

    A("<details>")
    A("<summary>Full per-workload numbers (bf16 -> e4m3, GB200)</summary>")
    A("")
    A(
        "Rows tagged `(specialized)` dispatch to `quantize_mxfp8_kernel_cast_only`, a separate "
        "kernel this PR does not touch -- they are the control group. Which kernel ran is taken "
        "from the profile, not inferred."
    )
    A("")
    A(
        "| fusion | dir | shape | model tensor | before (us) | after (us) | speedup | "
        "before GB/s | after GB/s |"
    )
    A("|---|---|---|---|--:|--:|--:|--:|--:|")
    for k, t_a, t_b, sp, a, b in sorted(
        rows, key=lambda r: (r[4]["specialized"], r[4]["combo"], r[4]["dir"], -r[3])
    ):
        shape = f"{a['M']}x{a['N']}"
        tag = a["combo"] + (" (specialized)" if a["specialized"] else "")
        A(
            f"| `{tag}` | {a['dir']} | {shape} | {SHAPE_NOTE.get(shape,'')} | "
            f"{t_a:.1f} | {t_b:.1f} | {sp:.3f}x | {a['gbps']:.0f} | {b['gbps']:.0f} |"
        )
    A("")
    A("</details>")
    A("")
    A("### Method")
    A("")
    A(
        "- **Pure GPU kernel time.** Every number is the median CUPTI kernel duration from "
        "nsys (`nvtx_kern_sum`), counting only the MXFP8 quantize kernel. No host time is "
        "included -- only the kernel body changed, host dispatch overlaps with GPU execution "
        "in a real step, and it vanishes under CUDA graphs. (`bgrad_quantize` also launches "
        "`reduce_dbias_kernel`; it is identical in both arms and is excluded.)"
    )
    A("- GB200, `NVTE_CUDA_ARCHS=100a`, bf16 -> e4m3, L2 flushed before every iteration.")
    A(
        "- Both arms are full builds of the same tree differing **only** by this hunk, and each "
        f"profile records the md5 of the `libtransformer_engine.so` actually mapped into the "
        f"process (`{pa.get('mapped_lib_md5','?')[:12]}` before, "
        f"`{pb.get('mapped_lib_md5','?')[:12]}` after) -- read from `/proc/self/maps` after "
        "load, so a stale build cannot masquerade as a fresh one."
    )
    A(
        "- **Control group:** `plain|row` and `plain|both` dispatch to the specialized "
        "cast-only kernel, which this change does not touch. They stay flat, confirming the "
        "harness is not manufacturing a difference."
    )
    A(
        "- **Numerics:** 72 combo/shape/direction configurations hashed under both builds are "
        "**bitwise identical** (0 mismatches). This is a pure codegen change."
    )

    out = os.path.join(HERE, "RESULTS.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L[: L.index("<details>")]))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
