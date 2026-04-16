#!/bin/bash
# Profile CuTeDSL MXFP8 quantization against the C++ reference with Nsight Systems.
#
# Default mode:  one .nsys-rep covering all shapes (combined timeline).
# Per-shape mode (--per-shape): one .nsys-rep per shape (cleaner isolated reports).
#
# Usage:
#   ./run_nsys_profile.sh                           # default preset, both dirs
#   ./run_nsys_profile.sh --preset square           # sweep 1k..16k square shapes
#   ./run_nsys_profile.sh --preset llm              # LLM-typical shapes
#   ./run_nsys_profile.sh --preset aspect           # tall/wide aspect ratios
#   ./run_nsys_profile.sh --shapes '8192,8192'      # single custom shape
#   ./run_nsys_profile.sh --per-shape --preset square
#   ./run_nsys_profile.sh --list-presets
#   WARMUP=20 ITERS=200 ./run_nsys_profile.sh --preset large
#
# Outputs:
#   profile/nsys_mxfp8_<TS>.nsys-rep        binary report (open with nsys-ui)
#   profile/nsys_mxfp8_<TS>.txt             stats summary + benchmark output
#   profile/nsys_mxfp8_<TS>.csv             per-shape timings

set -e

cd "$(dirname "$0")"
mkdir -p profile

# Pull --per-shape and --list-presets out before forwarding to Python
PER_SHAPE=0
PY_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --per-shape) PER_SHAPE=1; shift ;;
        --list-presets) python bench_mxfp8_cutedsl.py --list-presets; exit 0 ;;
        *) PY_ARGS+=("$1"); shift ;;
    esac
done

WARMUP=${WARMUP:-10}
ITERS=${ITERS:-100}

run_nsys() {
    local out="$1"; shift
    echo "==> nsys profile -> ${out}.nsys-rep"
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=false \
        --cpuctxsw=none \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --stats=true \
        --force-overwrite=true \
        --output="$out" \
        python bench_mxfp8_cutedsl.py \
            --warmup "$WARMUP" --iters "$ITERS" \
            --csv "${out}.csv" \
            "$@" \
        2>&1 | tee "${out}.txt"
}

if [[ $PER_SHAPE -eq 0 ]]; then
    # Single nsys invocation covering all shapes
    TS=$(date +"%Y%m%d-%H%M%S")
    OUT="profile/nsys_mxfp8_${TS}"
    run_nsys "$OUT" "${PY_ARGS[@]}"
    echo
    echo "==> Done"
    echo "    ${OUT}.nsys-rep   (open: nsys-ui ${OUT}.nsys-rep)"
    echo "    ${OUT}.txt        (stats + benchmark summary)"
    echo "    ${OUT}.csv        (per-shape timings)"
else
    # Extract shapes from --preset / --shapes and run nsys per shape
    SHAPES_TXT=$(python - <<'PY' "${PY_ARGS[@]}"
import argparse, sys
sys.path.insert(0, ".")
from bench_mxfp8_cutedsl import SHAPE_PRESETS, parse_shapes
p = argparse.ArgumentParser()
p.add_argument("--preset", default="default")
p.add_argument("--shapes")
p.add_argument("--direction", default="all")
# ignore the rest
args, _ = p.parse_known_args()
shapes = parse_shapes(args.shapes) if args.shapes else SHAPE_PRESETS[args.preset]
for m, n in shapes:
    print(f"{m},{n}")
PY
)

    for PAIR in $SHAPES_TXT; do
        M=${PAIR%,*}
        N=${PAIR#*,}
        TS=$(date +"%Y%m%d-%H%M%S")
        OUT="profile/nsys_mxfp8_${M}x${N}_${TS}"
        # Drop --preset/--shapes from PY_ARGS for the per-shape run
        FILTERED=()
        SKIP=0
        for arg in "${PY_ARGS[@]}"; do
            if [[ $SKIP -eq 1 ]]; then SKIP=0; continue; fi
            case "$arg" in
                --preset|--shapes) SKIP=1 ;;
                *) FILTERED+=("$arg") ;;
            esac
        done
        run_nsys "$OUT" --shapes "${M},${N}" "${FILTERED[@]}"
        # nsys writes its progress through, brief pause for clean tees
        sleep 1
    done
    echo
    echo "==> Done (per-shape mode)"
    ls -1t profile/nsys_mxfp8_*x*.nsys-rep 2>/dev/null | head -20
fi
