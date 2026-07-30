# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Render the generic-vs-specialized comparison as markdown, from pure GPU kernel time.

Colwise is omitted from the report: it has no specialized path, so there is
nothing to compare against (it only serves as an internal anchor, printed to
stdout as a sanity check).

Both ratio columns are oriented so that >1.00x always means faster.
"""

import json
import os
import re
import statistics

from nsys_load import load

HERE = os.path.dirname(os.path.abspath(__file__))
TITLE = {"both": "BIDIMENSIONAL", "row": "ROWWISE"}


def sass_by_direction():
    out = {}
    try:
        b = json.load(open(os.path.join(HERE, "sass", "base_sass.json")))["per"]
        f = json.load(open(os.path.join(HERE, "sass", "fix_sass.json")))["per"]
    except FileNotFoundError:
        return out
    for name in b:
        if not (re.search(r"quantize_mxfp8_kernelILb0ELb0ELb0E", name) and "__nv_bfloat16" in name):
            continue
        m = re.search(r"__nv_fp8_e4m3Lb(\d)ELb(\d)ELb0E", name)
        if not m:
            continue
        d = {("1", "0"): "row", ("1", "1"): "both"}.get(m.groups())
        if d:
            out[d] = (b[name]["generic"], f[name]["shared"])
    return out


def main():
    S, ps = load("spec_k")
    F, pf = load("fixgen_k")
    B, pb = load("basegen_k")

    L = []
    A = L.append
    A("## MXFP8 `quantize`: generic kernel vs specialized cast-only kernel")
    A("")
    A(
        "Three builds of the same tree, profiled on one GB200 (bf16 -> e4m3). All numbers are "
        "**pure GPU kernel time** (median CUPTI duration via nsys `nvtx_kern_sum`); no host "
        "time is included."
    )
    A("")
    A("| build | alignment fix | specialization | mapped `libtransformer_engine.so` md5 |")
    A("|---|---|---|---|")
    A(f"| **specialized** (stock) | yes | enabled | `{ps['mapped_lib_md5'][:12]}` |")
    A(f"| **generic+fix** | yes | disabled | `{pf['mapped_lib_md5'][:12]}` |")
    A(f"| **generic, no fix** | no (matches `main`) | disabled | `{pb['mapped_lib_md5'][:12]}` |")
    A("")
    A(
        "Specialization was disabled with a one-line `constexpr bool kAllowSpecialized = false` "
        "guard so that `plain` rowwise/bidimensional fall through to the generic kernel and can "
        "be compared head-to-head. That patch is experiment-only and is not part of the PR."
    )
    A("")
    A("Both ratio columns are oriented so **higher is faster**:")
    A("")
    A("- `fix vs nofix` -- what the alignment fix buys *within* the generic kernel.")
    A(
        "- `gen+fix vs spec` -- >1.00x means the fixed generic kernel is **faster** than the "
        "specialized cast-only kernel; <1.00x means it is still behind."
    )
    A("")

    stats = {}
    for d in ("both", "row"):
        keys = [k for k in S if f"|{d}|" in k]
        keys.sort(key=lambda k: -(S[k]["M"] * S[k]["N"]))
        A(f"### {TITLE[d]}")
        A("")
        A(
            "| shape | spec (us) | spec (GB/s) | gen+fix (us) | gen+fix (GB/s) | "
            "no-fix (us) | no-fix (GB/s) | fix vs nofix | gen+fix vs spec |"
        )
        A("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
        g, rt = [], []
        for k in keys:
            s, f, b = S[k], F[k], B[k]
            g.append(b["us"] / f["us"])
            rt.append(s["us"] / f["us"])
            A(
                f"| {s['M']}x{s['N']} | {s['us']:.1f} | {s['gbps']:.0f} | "
                f"{f['us']:.1f} | {f['gbps']:.0f} | {b['us']:.1f} | {b['gbps']:.0f} | "
                f"{b['us']/f['us']:.3f}x | {s['us']/f['us']:.3f}x |"
            )
        stats[d] = (statistics.median(g), statistics.median(rt), min(rt), max(rt))
        A(f"| **median** | | | | | | | **{stats[d][0]:.3f}x** | **{stats[d][1]:.3f}x** |")
        A("")

    mb, rb = stats["both"][1], stats["row"][1]
    verb = "faster than" if mb > 1 else "slower than"
    A(
        f"**Bidimensional:** the fixed generic kernel is {abs(100*(mb-1)):.1f}% {verb} the "
        f"specialized cast-only kernel (median {mb:.3f}x), and it wins at every shape except "
        f"2048x12288, where the two tie. The alignment fix itself accounts for "
        f"{stats['both'][0]:.3f}x of that within the generic kernel."
    )
    A("")
    A(
        f"**Rowwise:** the generic kernel is still ~{100*(1-rb):.0f}% slower (median "
        f"{rb:.3f}x), and the alignment fix does essentially nothing here "
        f"({stats['row'][0]:.3f}x)."
    )
    A("")

    # --- cold vs warm L2 -----------------------------------------------------
    try:
        W = {n: load(n + "_kw")[0] for n in ("spec", "fixgen")}
    except FileNotFoundError:
        W = None
    if W:
        A("### Cache regime: the bidimensional result depends on L2 residency")
        A("")
        A(
            "Identical measurement, the only difference being whether L2 is flushed between "
            "iterations. `gen+fix vs spec`, higher is faster:"
        )
        A("")
        A("| direction | shape | working set | cold L2 (from HBM) | warm L2 (resident) |")
        A("|---|---|--:|--:|--:|")
        for d in ("both", "row"):
            keys = [k for k in S if f"|{d}|" in k]
            keys.sort(key=lambda k: -(S[k]["M"] * S[k]["N"]))
            for k in keys:
                mb_ = S[k]["bytes"] / 1e6
                A(
                    f"| {TITLE[d]} | {S[k]['M']}x{S[k]['N']} | {mb_:.0f} MB | "
                    f"{S[k]['us']/F[k]['us']:.3f}x | "
                    f"{W['spec'][k]['us']/W['fixgen'][k]['us']:.3f}x |"
                )
        A("")
        A(
            "GB200's L2 is ~126 MB. The two smallest bidimensional shapes are the only ones "
            "whose working set fits, and they are exactly the ones where the ranking flips: "
            "served from L2 the kernel stops being bandwidth-bound, and the specialized "
            "kernel's larger 32x256 tile (half as many CTAs, so half the per-CTA prologue) "
            "wins on overhead. Streaming from HBM, the fixed generic kernel's tighter "
            "instruction stream wins instead. Rowwise does not flip -- the generic kernel "
            "trails in both regimes."
        )
        A("")

    sass = sass_by_direction()
    if sass:
        A("### Why rowwise is unaffected by the fix")
        A("")
        A("Memory ops in the generated SASS for the `plain` kernel, before -> after:")
        A("")
        A(
            "| direction | before (generic `LD.E`/`ST.E`) | after (`LDS`/`STS`) | "
            "measured `fix vs nofix` |"
        )
        A("|---|--:|--:|--:|")
        for d in ("both", "row"):
            if d in sass:
                A(f"| {TITLE[d]} | {sass[d][0]} | {sass[d][1]} | {stats[d][0]:.3f}x |")
        A("")
        A(
            "Rowwise-only stages roughly a quarter as much data through shared memory as the "
            "other layouts, so there is far less for the fix to convert -- which is exactly "
            "what the timings show."
        )
        A("")

    A("### Caveats")
    A("")
    A(
        "- Rowwise is a consistent loss for the generic kernel in **both** cache regimes. "
        "Bidimensional depends on L2 residency (see the section above): the generic kernel wins "
        "whenever the workload actually streams from HBM."
    )
    A(
        "- GB/s is effective bytes (inputs read + FP8 data + e8m0 scales written) divided by "
        "kernel time. Smaller shapes fit in GB200's L2, so those figures exceed HBM bandwidth "
        "and are not memory-bandwidth utilisation."
    )
    A(
        "- Only `plain` cast is covered: the specialized kernel does not exist for any fused "
        "(activation / dbias) variant."
    )
    A(
        "- Single dtype pair (bf16 -> e4m3) and 6 shapes; wider coverage would be needed before "
        "acting on the bidimensional result."
    )

    out = os.path.join(HERE, "SPECIALIZED_VS_GENERIC.md")
    open(out, "w").write("\n".join(L) + "\n")

    # colwise anchor: generic kernel in all three builds, so spec/gen+fix must be ~1.00x
    anchor = [S[k]["us"] / F[k]["us"] for k in S if "|col|" in k]
    print(f"colwise anchor (must be ~1.000x): median {statistics.median(anchor):.3f}x  "
          f"min {min(anchor):.3f}x  max {max(anchor):.3f}x")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
