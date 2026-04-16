#!/bin/bash
# Profile CuTeDSL MXFP8 quantization against the C++ reference with Nsight Systems.
#
# Generates:
#   profile/nsys_mxfp8_<timestamp>.nsys-rep   — binary report (open in nsys-ui)
#   profile/nsys_mxfp8_<timestamp>.txt        — stats summary
#
# Usage:
#   ./run_nsys_profile.sh                               # default sizes, both dirs
#   ./run_nsys_profile.sh --direction row               # rowwise only
#   ./run_nsys_profile.sh --shapes '8192,8192'          # single shape
#   WARMUP=5 ITERS=50 ./run_nsys_profile.sh             # custom iteration counts

set -e

cd "$(dirname "$0")"
mkdir -p profile

WARMUP=${WARMUP:-10}
ITERS=${ITERS:-100}
TS=$(date +"%Y%m%d-%H%M%S")
OUT="profile/nsys_mxfp8_${TS}"

# Pass-through arguments to the Python benchmark
PY_ARGS=("--warmup" "$WARMUP" "--iters" "$ITERS" "$@")

echo "==> Running nsys profile"
echo "    warmup=$WARMUP iters=$ITERS"
echo "    output=${OUT}.nsys-rep"
echo

nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=false \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --stats=true \
    --force-overwrite=true \
    --output="$OUT" \
    python bench_mxfp8_cutedsl.py "${PY_ARGS[@]}" \
    2>&1 | tee "${OUT}.txt"

echo
echo "==> Done"
echo "    binary report: ${OUT}.nsys-rep   (open with 'nsys-ui ${OUT}.nsys-rep')"
echo "    text output:   ${OUT}.txt"
echo
echo "==> Top CUDA kernels (from stats):"
grep -A 20 "CUDA Kernel Statistics" "${OUT}.txt" | head -30 || true
