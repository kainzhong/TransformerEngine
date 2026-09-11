#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Capture one MXFP8-quantize nsys perf summary for whatever build is currently
# installed. Run it once per build:
#
#   ./run_mxfp8_perf.sh some_label
#
# Every artifact records the branch, commit and extension build time it came
# from, so a label that turns out to be wrong is recoverable -- check the
# `build`/`built` columns in the CSV rather than trusting the filename.
#
# The sweep below is frozen on purpose: numbers from two builds only mean
# anything against each other if the workload matrix is identical.
set -euo pipefail

LABEL="${1:-run}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$HERE/mxfp8_perf}"

exec python "$HERE/profile_mxfp8_quantize.py" \
    --label "$LABEL" \
    --outdir "$OUTDIR" \
    --preset llm \
    --combos plain,gelu,silu,dgelu,dsilu,dbias,dbias_dgelu \
    --directions row,col,both \
    --in-dtype bf16 \
    --fp8 e4m3 \
    --warmup 10 \
    --iters 30
