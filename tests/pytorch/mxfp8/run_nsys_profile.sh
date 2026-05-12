#!/bin/bash
# Profile CuTeDSL vs C++ MXFP8 quantize kernel-only runtime per (shape, direction).
#
# For each config, wraps the benchmark in `nsys profile` and then extracts
# kernel-only GPU time from nsys's cuda_gpu_kern_sum report. This is the
# apples-to-apples kernel comparison — it excludes host-side launch overhead
# which `torch.cuda.Event` (what bench_mxfp8_cutedsl.py itself reports) rolls
# into its wall-clock timings.
#
# Note: bench_mxfp8_cutedsl.py now flushes L2 between iters by default, so
# the captured kernel launches see a cold L2. Pass --no-evict-l2 in
# EXTRA_ARGS to disable for warm-cache comparison.
#
# Usage:
#   ./run_nsys_profile.sh                           # default preset, all dirs
#   ./run_nsys_profile.sh --preset square
#   ./run_nsys_profile.sh --shapes '8192,8192'
#   ./run_nsys_profile.sh --direction row           # just rowwise
#   ./run_nsys_profile.sh --list-presets
#   WARMUP=20 ITERS=200 ./run_nsys_profile.sh --preset large
#
# Outputs:
#   profile/nsys_kernel_time/nsys_<shape>_<dir>_<TS>.nsys-rep
#   profile/nsys_kernel_time/nsys_<shape>_<dir>_<TS>.stdout
#   Plus a summary table printed to stdout at the end.

set -e
cd "$(dirname "$0")"
mkdir -p profile/nsys_kernel_time

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
        --preset) PRESET="$2"; shift 2 ;;
        --shapes) SHAPES_ARG="$2"; shift 2 ;;
        --direction) DIR_ARG="$2"; shift 2 ;;
        --combo|--combos) COMBOS_ARG="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
IFS=',' read -r -a COMBOS <<< "$COMBOS_ARG"

WARMUP=${WARMUP:-10}
ITERS=${ITERS:-100}

# --- expand shapes ---
if [[ -n "$SHAPES_ARG" ]]; then
    SHAPES=$(echo "$SHAPES_ARG" | tr ';' '\n')
else
    SHAPES=$(python - <<PY
from bench_mxfp8_cutedsl import SHAPE_PRESETS
for m, n in SHAPE_PRESETS["$PRESET"]:
    print(f"{m},{n}")
PY
)
fi

# --- expand directions ---
case "$DIR_ARG" in
    all)  DIRS=("row" "col" "both") ;;
    row)  DIRS=("row") ;;
    col)  DIRS=("col") ;;
    both) DIRS=("both") ;;
    *) echo "invalid --direction: $DIR_ARG (want row|col|both|all)"; exit 1 ;;
esac

# --- run nsys per (shape, dir) and collect kernel-only stats ---
RESULTS_FILE=$(mktemp)
trap 'rm -f "$RESULTS_FILE"' EXIT

for COMBO in "${COMBOS[@]}"; do
for SHAPE_PAIR in $SHAPES; do
    M=${SHAPE_PAIR%,*}
    N=${SHAPE_PAIR#*,}
    for DIR in "${DIRS[@]}"; do
        TS=$(date +"%Y%m%d-%H%M%S")
        OUT="profile/nsys_kernel_time/nsys_${COMBO}_${M}x${N}_${DIR}_${TS}"
        echo "==> nsys: ${COMBO} ${M}x${N} ${DIR}"

        if ! nsys profile \
            --trace=cuda,nvtx \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --stats=false \
            --force-overwrite=true \
            --output="$OUT" \
            python bench_mxfp8_cutedsl.py \
                --warmup "$WARMUP" --iters "$ITERS" \
                --shapes "${M},${N}" --direction "$DIR" \
                --combo "$COMBO" \
                "${EXTRA_ARGS[@]}" \
            > "${OUT}.stdout" 2>&1
        then
            echo "   FAILED — see ${OUT}.stdout"
            echo "${COMBO} ${M}x${N} ${DIR} 0 0 0 0 0" >> "$RESULTS_FILE"
            continue
        fi

        # Bytes moved per single-kernel launch. Activation/dbias variants
        # add an extra act_input read (bf16); other terms identical.
        EXTRA_IN=0
        case "$COMBO" in
            dgelu|dsilu|drelu|dbias_dgelu|dbias_dsilu|dbias_drelu)
                EXTRA_IN=1 ;;
        esac
        if [[ "$DIR" == "both" ]]; then
            BYTES=$(awk -v m="$M" -v n="$N" -v ei="$EXTRA_IN" \
                'BEGIN{printf "%.0f", (2 + 2*ei)*m*n + 2*m*n + 2*(m*n/32)}')
        else
            BYTES=$(awk -v m="$M" -v n="$N" -v ei="$EXTRA_IN" \
                'BEGIN{printf "%.0f", (2 + 2*ei)*m*n + m*n + (m*n/32)}')
        fi

        # Extract kernel-only avg/instances via cuda_gpu_kern_sum CSV.
        # DSL kernel name contains "kernel_cutlass_kernel_quantize_mxfp8".
        # The TE C++ side's main quantize-cast kernel is the largest non-DSL
        # kernel containing "quantize" (covers plain quantize_mxfp8_kernel
        # template variants for IS_ACT/IS_DACT/IS_DBIAS, plus the activation
        # entry kernels). Small util kernels (reduce_dbias, torch.sum CUB
        # reductions, RNG) are ignored.
        PARSED=$(python - "$OUT" <<'PY'
import csv, io, subprocess, sys
rep = sys.argv[1] + ".nsys-rep"
out = subprocess.run(
    ["nsys", "stats", "--report", "cuda_gpu_kern_sum",
     "--format", "csv", rep],
    capture_output=True, text=True,
)
dsl_avg = cpp_avg = None
dsl_inst = cpp_inst = 0
cpp_total_best = -1.0
for row in csv.reader(io.StringIO(out.stdout)):
    if len(row) < 9:
        continue
    try:
        total_ns = float(row[1])
        inst = int(row[2])
        avg_ns = float(row[3])
    except ValueError:
        continue
    name = row[-1]
    if "kernel_cutlass" in name or "cutedsl_alt" in name:
        if dsl_avg is None:
            dsl_avg, dsl_inst = avg_ns, inst
    else:
        # Pick the dominant TE quantize/cast kernel by total time.
        if "quantize_mxfp8_kernel" in name and total_ns > cpp_total_best:
            cpp_total_best = total_ns
            cpp_avg, cpp_inst = avg_ns, inst
print(f"{dsl_avg or 0} {cpp_avg or 0} {dsl_inst} {cpp_inst}")
PY
        )
        echo "${COMBO} ${M}x${N} ${DIR} ${PARSED} ${BYTES}" >> "$RESULTS_FILE"
    done
done
done

# --- print summary ---
echo
echo "Kernel-only runtime (nsys cuda_gpu_kern_sum, WARMUP=${WARMUP} ITERS=${ITERS})"
echo "============================================================================================================="
printf "%-14s  %-14s  %-5s  %9s  %9s  %9s  %9s  %8s\n" \
    "combo" "shape" "dir" "DSL us" "C++ us" "DSL GB/s" "C++ GB/s" "DSL/C++"
printf -- "--------------  --------------  -----  ---------  ---------  ---------  ---------  --------\n"

while read -r COMBO SHAPE DIR DSL_NS CPP_NS DI CI BYTES; do
    DSL_US=$(awk -v v="$DSL_NS" 'BEGIN{printf "%.2f", v/1000.0}')
    CPP_US=$(awk -v v="$CPP_NS" 'BEGIN{printf "%.2f", v/1000.0}')
    DSL_GBPS=$(awk -v b="$BYTES" -v t="$DSL_NS" 'BEGIN{if(t>0) printf "%.1f", b/t; else print "n/a"}')
    CPP_GBPS=$(awk -v b="$BYTES" -v t="$CPP_NS" 'BEGIN{if(t>0) printf "%.1f", b/t; else print "n/a"}')
    # speedup = C++ time / DSL time  (>1.0 = DSL faster).
    if awk -v d="$DSL_NS" 'BEGIN{exit (d>0)?0:1}'; then
        SPEEDUP=$(awk -v d="$DSL_NS" -v c="$CPP_NS" 'BEGIN{printf "%.3fx", c/d}')
    else
        SPEEDUP="n/a"
    fi
    printf "%-14s  %-14s  %-5s  %9s  %9s  %9s  %9s  %8s\n" \
        "$COMBO" "$SHAPE" "$DIR" "$DSL_US" "$CPP_US" "$DSL_GBPS" "$CPP_GBPS" "$SPEEDUP"
done < "$RESULTS_FILE"

echo
echo "==> Reports in profile/nsys_kernel_time/"
