#!/bin/bash
# Profile the CUDA C++ and CuTeDSL MXFP8 kernels with Nsight Compute.
#
# Each ncu invocation profiles exactly ONE (combo, shape, direction) config and
# captures one C++ ref launch + one CuTeDSL launch (warmup happens outside
# torch.cuda.profiler.start/stop, so only the measurement calls are
# captured).  We launch ncu once per config because `--set full` collects
# every counter — putting many configs behind a single ncu makes the
# replay phase intractably long.
#
# bench_for_ncu.py issues exactly ONE kernel launch per profiled section,
# since ncu replays the kernel internally to collect each counter group —
# a second launch would just be wasted setup work.
#
# bench_for_ncu.py also flushes L2 once before each profiled kernel launch
# (and drains the evict kernel before profiler.start), so NCU captures the
# target kernel running against a cold L2 — matching production cold-cache
# latency.
#
# Output directory: <repo-root>/profile/
#
# Usage:
#   ./run_ncu_profile.sh                              # preset=default, dir=all, combo=plain
#   ./run_ncu_profile.sh --preset square
#   ./run_ncu_profile.sh --preset full                # all 13 sweep shapes
#   ./run_ncu_profile.sh --direction row
#   ./run_ncu_profile.sh --shapes '8192,8192'         # ad-hoc single shape
#   ./run_ncu_profile.sh --combo plain,dgelu          # multiple combos
#   ./run_ncu_profile.sh --list-presets
#   WARMUP=50 ./run_ncu_profile.sh
#
# Outputs (timestamped, per config):
#   ncu_mxfp8_<combo>_<M>x<N>_<dir>_<TS>.ncu-rep    open with: ncu-ui <file>
#   ncu_mxfp8_<combo>_<M>x<N>_<dir>_<TS>.txt        stdout + stderr of the run

set -e

cd "$(dirname "$0")"
PROFILE_DIR="$(git rev-parse --show-toplevel)/profile"
mkdir -p "$PROFILE_DIR"

# Gate for TE's fast-path MXFP8 kernel.  Without this env var,
# `specialized::hasSpec<...>()` returns false and TE dispatches to the
# generic `quantize_mxfp8_kernel` instead of `specialized::
# quantize_mxfp8_kernel_cast_only` — which is the TMA path we actually
# want to compare the CuTeDSL kernel against.  See
# transformer_engine/common/cast/mxfp8/specialized/quantize_mxfp8.cuh.
export ENABLE_CAST_ONLY=${ENABLE_CAST_ONLY:-1}

# --- arg parsing ---
PRESET="default"
SHAPES_ARG=""
DIR_ARG="all"
COMBOS_ARG="plain"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list-presets)
            python bench_mxfp8_cutedsl.py --list-presets
            exit 0 ;;
        --preset)         PRESET="$2"; shift 2 ;;
        --shapes)         SHAPES_ARG="$2"; shift 2 ;;
        --direction)      DIR_ARG="$2"; shift 2 ;;
        --combo|--combos) COMBOS_ARG="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,33p' "$0"
            exit 0
            ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
IFS=',' read -r -a COMBOS <<< "$COMBOS_ARG"

WARMUP=${WARMUP:-20}

# Resolve direction list.
case "$DIR_ARG" in
    all)  DIRS=("row" "col" "both") ;;
    row)  DIRS=("row") ;;
    col)  DIRS=("col") ;;
    both) DIRS=("both") ;;
    *) echo "--direction must be row|col|both|all"; exit 1 ;;
esac

# Resolve shape list (from --shapes or SHAPE_PRESETS[preset]).
if [[ -n "$SHAPES_ARG" ]]; then
    SHAPES=$(echo "$SHAPES_ARG" | tr ';' '\n')
else
    SHAPES=$(python - <<PY
from bench_mxfp8_cutedsl import SHAPE_PRESETS
preset = "$PRESET"
if preset not in SHAPE_PRESETS:
    import sys
    sys.exit(f"unknown preset {preset!r}; choices: {sorted(SHAPE_PRESETS)}")
for m, n in SHAPE_PRESETS[preset]:
    print(f"{m},{n}")
PY
)
fi

echo "==> Profile dir: ${PROFILE_DIR}"
echo "==> Preset: ${PRESET}   direction: ${DIR_ARG}   combos: ${COMBOS_ARG}   warmup: ${WARMUP}"
echo "==> Shapes:"
echo "$SHAPES" | sed 's/^/    /'
echo

# Loop: one ncu invocation per (combo, shape, direction).
for COMBO in "${COMBOS[@]}"; do
for SHAPE_PAIR in $SHAPES; do
    M="${SHAPE_PAIR%,*}"
    N="${SHAPE_PAIR#*,}"
    for D in "${DIRS[@]}"; do
        TS=$(date +"%Y%m%d-%H%M%S")
        OUT="${PROFILE_DIR}/ncu_mxfp8_${COMBO}_${M}x${N}_${D}_${TS}"
        echo "==> ${COMBO} ${M}x${N} ${D}  ->  ${OUT}.ncu-rep"

        ncu \
            --set full \
            --target-processes all \
            --profile-from-start off \
            --nvtx \
            --force-overwrite \
            -o "$OUT" \
            python bench_for_ncu.py \
                --shapes "${M},${N}" \
                --direction "${D}" \
                --combo "${COMBO}" \
                --warmup "${WARMUP}" \
                "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${OUT}.txt"

        sleep 1
    done
done
done

echo
echo "==> Done"
echo "    Latest reports:"
ls -1t "${PROFILE_DIR}"/ncu_mxfp8_*.ncu-rep 2>/dev/null | head -10
