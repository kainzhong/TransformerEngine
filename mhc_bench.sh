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

for config in "${CONFIGS[@]}"; do
    read -r B T C <<< "$config"
    OUTFILE="$SCRIPT_DIR/nsys_mhc_B${B}_T${T}_C${C}.txt"

    echo "=== B=$B T=$T C=$C ===" | tee "$OUTFILE"

    nsys profile \
        --trace=cuda \
        --cuda-memory-usage=false \
        --cpuctxsw=none \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --stats=true \
        --force-overwrite=true \
        --output="$TMPBASE" \
        python "$BENCH" \
            --operation all \
            --B "$B" --T "$T" --C "$C" \
            --warmup 3 \
            --iters 5 \
        2>&1 | tee -a "$OUTFILE"

    # Discard the nsys report files
    rm -f "${TMPBASE}.nsys-rep" "${TMPBASE}.sqlite"

    echo "" >> "$OUTFILE"
    echo "Written: $OUTFILE"
done
