#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Final A/B for the five *reachable* SHMEM-provenance fixes, cold and warm L2.
#
#   before : base_libte.so     main, no fixes
#   after  : helper_libte.so   all fixes, via the shared align_up() helper
#
#   cast/mxfp8/quantize_mxfp8.cuh                generic quantize kernel
#   cast/mxfp8/specialized/quantize_mxfp8.cuh    bidimensional cast-only kernel
#   cast/mxfp8/gated_mxfp8.cuh                   MXFP8 gated (swiglu/geglu)
#   cast/nvfp4/quantize_transpose_nvfp4.cuh      NVFP4 2D quantize-transpose
#   cast/fp8/gated_fp8.cuh                       FP8 gated (swiglu/geglu)
#
# All numbers are pure GPU kernel time (CUPTI, via nsys nvtx_kern_sum).
#
# BOTH arms are always re-run together in one invocation: kernel times are only
# comparable within a single node, and every profile now records the hostname so
# a cross-node comparison is detectable after the fact.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
PROF="$HERE/../profile_mxfp8_quantize.py"
OUT="$HERE/nsys"
LIVE="$REPO/libtransformer_engine.so"
mkdir -p "$OUT"

MXFP8_COMBOS=plain,gelu,dgelu,dsilu,dbias,dbias_dgelu,swiglu,dswiglu,geglu

run() {  # lib label bench [extra...]
    cp "$HERE/$1.so" "$LIVE"
    echo "=== $2 ($1) ==="
    local bench=()
    [ -n "$3" ] && bench=(--bench "$3")
    ( cd "$REPO" && python "$PROF" "${bench[@]}" --label "$2" --outdir "$OUT" \
        --warmup 10 --iters 30 "${@:4}" )
}

for arm in base helper; do
    lib="${arm}_libte"
    run "$lib" "f2_${arm}_cold"   "" --preset llm --combos "$MXFP8_COMBOS" --directions row,col,both
    run "$lib" "f2_${arm}_warm"   "" --preset llm --combos "$MXFP8_COMBOS" --directions row,col,both --no-evict
    run "$lib" "f2nv_${arm}_cold" bench_nvfp4_quantize.py --directions row,col,both --modes 2d
    run "$lib" "f2nv_${arm}_warm" bench_nvfp4_quantize.py --directions row,col,both --modes 2d --no-evict
    run "$lib" "f2f8_${arm}_cold" bench_fp8_gated_quantize.py
    run "$lib" "f2f8_${arm}_warm" bench_fp8_gated_quantize.py --no-evict
done

cp "$HERE/helper_libte.so" "$LIVE"

echo
echo "=== library and host recorded by each profile ==="
OUT="$OUT" python3 - <<'PY'
import glob, json, os
for f in sorted(glob.glob(os.path.join(os.environ["OUT"], "f2*_manifest.json"))):
    p = json.load(open(f))["provenance"]
    print(f"  {os.path.basename(f).replace('_manifest.json',''):18} "
          f"{p.get('mapped_lib_md5','?')[:12]}  {p.get('host','?')}")
PY
