#!/bin/bash
# Benchmark mhc kernels with nsys across common LLM training shapes.
# Stats are printed to stdout and redirected to per-config log files.
# nsys report files are discarded to keep things clean.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$SCRIPT_DIR/mhc_bench.py"

# Common LLM training shapes: "B T C"
#   LLaMA-7B  / Mistral-7B:  hidden=4096
#   LLaMA-13B:               hidden=5120
#   LLaMA-70B / Qwen-72B:    hidden=8192
# Batch sizes are per-GPU micro-batch sizes typical in training runs.
CONFIGS=(
    "1  2048 4096"
    "2  2048 4096"
    "4  2048 4096"
    "1  4096 4096"
    "2  4096 4096"
    "1  2048 5120"
    "2  2048 5120"
    "1  2048 8192"
    "2  2048 8192"
    "1  4096 8192"
)

TMPBASE=$(mktemp -u /tmp/nsys_mhc_XXXXXX)

# Run a command, stream+append output to OUTFILE, retry on ldconfig SIGSEGV.
# Usage: run_with_retry <outfile> <cmd> [args...]
run_with_retry() {
    local outfile="$1"; shift
    local max_attempts=5
    local attempt=1
    local tmpout exit_code
    tmpout=$(mktemp)

    while [ $attempt -le $max_attempts ]; do
        set +e
        "$@" >"$tmpout" 2>&1
        exit_code=$?
        set -e
        tee -a "$outfile" <"$tmpout"

        if [ $exit_code -eq 0 ]; then
            rm -f "$tmpout"
            return 0
        fi

        if grep -qF "SIGSEGV" "$tmpout" && grep -qF "ldconfig" "$tmpout"; then
            echo ">>> ldconfig SIGSEGV detected, retrying (attempt $attempt/$max_attempts)..." | tee -a "$outfile"
            attempt=$((attempt + 1))
            continue
        fi

        rm -f "$tmpout"
        return $exit_code
    done

    rm -f "$tmpout"
    return $exit_code
}

for config in "${CONFIGS[@]}"; do
    read -r B T C <<< "$config"
    OUTFILE="$SCRIPT_DIR/profile/nsys_mhc_B${B}_T${T}_C${C}.txt"

    echo "=== B=$B T=$T C=$C ===" | tee "$OUTFILE"
    run_with_retry "$OUTFILE" nsys profile \
        --trace=cuda,nvtx \
        --cuda-memory-usage=false \
        --cpuctxsw=none \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --stats=true \
        --force-overwrite=true \
        python "$BENCH" \
            --operation all \
            --B "$B" --T "$T" --C "$C" \
            --warmup 3 \
            --iters 1

    OUTFILE="$SCRIPT_DIR/profile/ncu_mhc_B${B}_T${T}_C${C}.txt"
    run_with_retry "$OUTFILE" ncu \
        --clock-control none --cache-control none --metrics gpu__time_duration.sum \
        --kernel-name regex:"_mhc.*|_ct.*" --profile-from-start off \
        python "$BENCH" --B "$B" --T "$T" --C "$C" --operation all --warmup 3 --iters 1

    echo "" >> "$OUTFILE"
    echo "Written: $OUTFILE"
done
