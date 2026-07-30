#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Reproduce RESULTS.md and SPECIALIZED_VS_GENERIC.md end to end.
#
#   ./run_nsys_all.sh          # profile all builds  (~20 min)
#   python make_report.py      # -> RESULTS.md
#   python gen_spec_report.py  # -> SPECIALIZED_VS_GENERIC.md
#
# Everything measured here is PURE GPU KERNEL TIME: the median CUPTI kernel
# duration from nsys's nvtx_kern_sum, counting only the MXFP8 quantize kernel.
# Wall-clock is deliberately not used -- only the kernel body changed, host
# dispatch overlaps with GPU execution in a real step, and it disappears
# entirely under CUDA graphs.
#
# Requires four prebuilt libraries in this directory (each ~150 MB, gitignored):
#
#   fix_libte.so          this branch                      (alignment fix, specialization on)
#   base_libte.so         main                             (no fix,        specialization on)
#   fix_nospec_libte.so   this branch + kAllowSpecialized=false
#   base_nospec_libte.so  main        + kAllowSpecialized=false
#
# The *_nospec pair is an experiment-only build: a `constexpr bool
# kAllowSpecialized = false` guard added to the dispatch `if` in
# quantize_mxfp8.cuh so that plain rowwise/bidimensional fall through to the
# generic kernel and can be compared against the specialized cast-only kernel.
# That guard is NOT part of the PR. Build each arm with:
#
#   export NVTE_CUDA_ARCHS=100a NVTE_FRAMEWORK=pytorch
#   pip install -e . --no-build-isolation -v
#   cp libtransformer_engine.so tests/pytorch/mxfp8/proof/<name>.so
#
# Every profile records the md5 of the libtransformer_engine.so actually mapped
# into the process (read from /proc/self/maps after load) -- printed at the end.
# That check is load-bearing: this box also carries a prebuilt TE in
# dist-packages, and an earlier A/B silently compared one build against itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
PROF="$HERE/../profile_mxfp8_quantize.py"
OUT="$HERE/nsys"
LIVE="$REPO/libtransformer_engine.so"
mkdir -p "$OUT"

COMBOS=plain,gelu,dgelu,dsilu,dbias,dbias_dgelu

run() {
    local lib="$1" label="$2"; shift 2
    [ -f "$HERE/${lib}.so" ] || { echo "missing $HERE/${lib}.so -- see header" >&2; exit 1; }
    cp "$HERE/${lib}.so" "$LIVE"
    echo "=================== $label  ($lib) ==================="
    ( cd "$REPO" && python "$PROF" --label "$label" --outdir "$OUT" "$@" )
}

# --- the PR A/B: every fusion, every direction, cold L2 ----------------------
run base_libte base_k --preset llm --combos "$COMBOS" --directions row,col,both --warmup 10 --iters 30
run fix_libte  fix_k  --preset llm --combos "$COMBOS" --directions row,col,both --warmup 10 --iters 30

# --- generic vs specialized, cold L2 (col kept as an anchor) -----------------
run fix_libte         spec_k    --preset llm --combos plain --directions row,col,both --warmup 10 --iters 30
run fix_nospec_libte  fixgen_k  --preset llm --combos plain --directions row,col,both --warmup 10 --iters 30
run base_nospec_libte basegen_k --preset llm --combos plain --directions row,col,both --warmup 10 --iters 30

# --- same again warm, to show the bidimensional result depends on L2 residency
run fix_libte         spec_kw    --preset llm --combos plain --directions row,col,both --warmup 10 --iters 30 --no-evict
run fix_nospec_libte  fixgen_kw  --preset llm --combos plain --directions row,col,both --warmup 10 --iters 30 --no-evict
run base_nospec_libte basegen_kw --preset llm --combos plain --directions row,col,both --warmup 10 --iters 30 --no-evict

cp "$HERE/fix_libte.so" "$LIVE"

echo
echo "=== library actually mapped in each profile ==="
OUT="$OUT" python - <<'PY'
import glob, json, os
for f in sorted(glob.glob(os.path.join(os.environ["OUT"], "*_manifest.json"))):
    p = json.load(open(f))["provenance"]
    print(f"  {os.path.basename(f).replace('_manifest.json',''):12} {p.get('mapped_lib_md5')}")
PY
