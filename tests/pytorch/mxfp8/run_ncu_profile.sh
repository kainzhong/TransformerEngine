#!/bin/bash
# Profile the CUDA C++ and CuTeDSL MXFP8 kernels with Nsight Compute.
#
# Each ncu invocation profiles exactly ONE (shape, direction) config and
# captures one C++ ref launch + one CuTeDSL launch (warmup happens outside
# torch.cuda.profiler.start/stop, so only the measurement calls are
# captured).  We launch ncu once per config because `--set full` collects
# every counter — putting many configs behind a single ncu makes the
# replay phase intractably long.
#
# bench_for_ncu.py flushes L2 once before each profiled kernel launch (and
# drains the evict kernel before profiler.start), so NCU captures the target
# kernel running against a cold L2 — matching production cold-cache latency.
#
# Output directory: <repo-root>/profile/
#
# Usage:
#   ./run_ncu_profile.sh                        # preset=default, direction=all
#   ./run_ncu_profile.sh --preset square
#   ./run_ncu_profile.sh --preset full          # all 13 sweep shapes
#   ./run_ncu_profile.sh --direction row
#   ./run_ncu_profile.sh --shapes '8192,8192'   # ad-hoc single shape
#   WARMUP=50 ./run_ncu_profile.sh
#
# Outputs (timestamped, per config):
#   ncu_mxfp8_<M>x<N>_<dir>_<TS>.ncu-rep    open with: ncu-ui <file>
#   ncu_mxfp8_<M>x<N>_<dir>_<TS>.txt        stdout + stderr of the run

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

WARMUP=${WARMUP:-20}
PRESET=${PRESET:-default}
DIR=${DIR:-all}
SHAPES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)    PRESET="$2"; shift 2 ;;
        --direction) DIR="$2"; shift 2 ;;
        --warmup)    WARMUP="$2"; shift 2 ;;
        --shapes)    SHAPES="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,28p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Resolve direction list.
case "$DIR" in
    all)  DIRS=("row" "col" "both") ;;
    row)  DIRS=("row") ;;
    col)  DIRS=("col") ;;
    both) DIRS=("both") ;;
    *) echo "--direction must be row|col|both|all"; exit 1 ;;
esac

# Resolve shape list (from --shapes or SHAPE_PRESETS[preset]).
if [[ -n "$SHAPES" ]]; then
    SHAPE_LIST="$SHAPES"
else
    SHAPE_LIST=$(python - "$PRESET" <<'PY'
import sys
from bench_mxfp8_cutedsl import SHAPE_PRESETS
preset = sys.argv[1]
if preset not in SHAPE_PRESETS:
    sys.exit(f"unknown preset {preset!r}; "
             f"choices: {sorted(SHAPE_PRESETS)}")
print(";".join(f"{m},{n}" for m, n in SHAPE_PRESETS[preset]))
PY
)
fi

echo "==> Profile dir: ${PROFILE_DIR}"
echo "==> Preset: ${PRESET}   direction: ${DIR}   warmup: ${WARMUP}"
echo "==> Shapes: ${SHAPE_LIST}"
echo

# Loop: one ncu invocation per (shape, direction).
IFS=';' read -ra SHAPE_ARR <<< "$SHAPE_LIST"
for SHAPE in "${SHAPE_ARR[@]}"; do
    M="${SHAPE%,*}"
    N="${SHAPE#*,}"
    for D in "${DIRS[@]}"; do
        TS=$(date +"%Y%m%d-%H%M%S")
        OUT="${PROFILE_DIR}/ncu_mxfp8_${M}x${N}_${D}_${TS}"
        echo "==> ${M}x${N} ${D}  ->  ${OUT}.ncu-rep"

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
                --warmup "${WARMUP}" \
            2>&1 | tee "${OUT}.txt"

        sleep 1
    done
done

echo
echo "==> Done"
echo "    Latest reports:"
ls -1t "${PROFILE_DIR}"/ncu_mxfp8_*.ncu-rep 2>/dev/null | head -10
