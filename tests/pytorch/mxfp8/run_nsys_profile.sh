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
#   ./run_nsys_profile.sh --hybrid --shapes '4096,4096' --direction row
#       Bench the fused hybrid MXFP8+NVFP4 kernel (bench_hybrid_cutedsl.py).
#       DSL = fused kernel; "C++" column = TE MXFP8 + NVFP4 quantizers run
#       SEPARATELY (sum of all transformer_engine kernels). --direction here
#       selects the MXFP8 direction (NVFP4 goes in the other); no 'both'.
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
COMBOS_ARG=""
HYBRID=""                       # --hybrid: bench the fused MXFP8+NVFP4 kernel
BENCH_SCRIPT="bench_mxfp8_cutedsl.py"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list-presets)
            python "$BENCH_SCRIPT" --list-presets
            exit 0 ;;
        --hybrid) HYBRID=1; BENCH_SCRIPT="bench_hybrid_cutedsl.py"; shift ;;
        --preset) PRESET="$2"; shift 2 ;;
        --shapes) SHAPES_ARG="$2"; shift 2 ;;
        --direction) DIR_ARG="$2"; shift 2 ;;
        --combo|--combos) COMBOS_ARG="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
IFS=',' read -r -a COMBOS <<< "$COMBOS_ARG"
# Hybrid bench has no --combo knob (it's MXFP8-one-dir + NVFP4-other-dir);
# use a single pseudo-combo "hybrid" purely for output-file naming.
if [[ -n "$HYBRID" ]]; then
    COMBOS=("hybrid")
fi

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
# The hybrid kernel does MXFP8 in one direction + NVFP4 in the other, so the
# DIR here means "the MXFP8 direction"; there is no "both".
if [[ -n "$HYBRID" ]]; then
    case "$DIR_ARG" in
        all)  DIRS=("row" "col") ;;
        both) echo "--hybrid has no 'both' (MXFP8 one dir + NVFP4 other)"; exit 1 ;;
    esac
fi

# --- run nsys per (shape, dir) and collect kernel-only stats ---
RESULTS_FILE=$(mktemp)
trap 'rm -f "$RESULTS_FILE"' EXIT
# Track the files generated per run so we can list them at the end.
GENERATED_FILES=()

for COMBO in "${COMBOS[@]}"; do
for SHAPE_PAIR in $SHAPES; do
    M=${SHAPE_PAIR%,*}
    N=${SHAPE_PAIR#*,}
    for DIR in "${DIRS[@]}"; do
        TS=$(date +"%Y%m%d-%H%M%S")
        OUT="profile/nsys_kernel_time/nsys_${COMBO}_${M}x${N}_${DIR}_${TS}"
        echo "==> nsys: ${COMBO} ${M}x${N} ${DIR}"

        # Build bench args. The hybrid bench takes --mxfp8-dir and has no --combo.
        BENCH_ARGS=(--warmup "$WARMUP" --iters "$ITERS" --shapes "${M},${N}")
        if [[ -n "$HYBRID" ]]; then
            BENCH_ARGS+=(--mxfp8-dir "$DIR")
        else
            BENCH_ARGS+=(--direction "$DIR" --combo "$COMBO")
        fi
        BENCH_ARGS+=("${EXTRA_ARGS[@]}")

        if ! nsys profile \
            --trace=cuda,nvtx \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --stats=true \
            --resolve-symbols=false \
            --force-overwrite=true \
            --output="$OUT" \
            python "$BENCH_SCRIPT" "${BENCH_ARGS[@]}" \
            > "${OUT}.stdout" 2>&1
        then
            echo "   FAILED — see ${OUT}.stdout"
            # Fields: combo shape dir DSL_kern CPP_kern DI CI BYTES CPP_BYTES DSL_wall CPP_wall
            echo "${COMBO} ${M}x${N} ${DIR} 0 0 0 0 0 0 0 0" >> "$RESULTS_FILE"
            continue
        fi

        # Bytes moved per timed iteration → GB/s = bytes / kernel_ns.
        # MXFP8 bench: DSL and C++ move the same bytes (BYTES == CPP_BYTES).
        # --hybrid: the fused DSL kernel reads the input ONCE; the separate TE
        # baseline (MXFP8 + NVFP4 quantizers) reads it TWICE — so they differ.
        # SOL now includes the extra NVFP4 outputs (data m*n/2 + scale m*n/16)
        # on top of the MXFP8 outputs (data m*n + scale m*n/32).
        if [[ -n "$HYBRID" ]]; then
            OUTB="m*n + (m*n/32) + (m*n/2) + (m*n/16)"  # mxfp8 data+scale + nvfp4 data+scale
            BYTES=$(awk -v m="$M" -v n="$N" "BEGIN{printf \"%.0f\", 2*m*n + ${OUTB}}")
            CPP_BYTES=$(awk -v m="$M" -v n="$N" "BEGIN{printf \"%.0f\", 4*m*n + ${OUTB}}")
        else
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
            CPP_BYTES="$BYTES"
        fi

        # Extract kernel-only avg/instances via cuda_gpu_kern_sum CSV.
        # DSL kernel name contains "kernel_cutlass_kernel_quantize_mxfp8".
        # The TE C++ side's main quantize-cast kernel is the largest non-DSL
        # kernel containing "quantize" (covers plain quantize_mxfp8_kernel
        # template variants for IS_ACT/IS_DACT/IS_DBIAS, plus the activation
        # entry kernels). Small util kernels (reduce_dbias, torch.sum CUB
        # reductions, RNG) are ignored.
        PARSED=$(python - "$OUT" "${HYBRID:-0}" <<'PY'
import csv, io, subprocess, sys
rep = sys.argv[1] + ".nsys-rep"
hybrid = sys.argv[2] == "1"
out = subprocess.run(
    ["nsys", "stats", "--report", "cuda_gpu_kern_sum",
     "--format", "csv", "--force-export=true", rep],
    capture_output=True, text=True,
)
dsl_avg = cpp_avg = None
dsl_inst = cpp_inst = 0
cpp_total_best = -1.0
cpp_sum = 0.0  # hybrid: sum of all TE-namespace baseline kernels' per-iter avg
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
    elif hybrid:
        # Baseline = TE MXFP8 + TE NVFP4 quantizers run separately. Sum every
        # transformer_engine-namespace kernel's per-iter avg: NVFP4
        # block_scaled_1d_cast_transpose, MXFP8 quantize_mxfp8_kernel, and the
        # NVFP4 amax_kernel / zero_amax_kernel. (torch at::native reductions
        # from the one-off s_enc precompute are not in this namespace.)
        if "transformer_engine" in name:
            cpp_sum += avg_ns
            cpp_inst = max(cpp_inst, inst)
    else:
        # MXFP8 bench: the dominant TE quantize/cast kernel by total time.
        if "quantize_mxfp8_kernel" in name and total_ns > cpp_total_best:
            cpp_total_best = total_ns
            cpp_avg, cpp_inst = avg_ns, inst
if hybrid:
    cpp_avg = cpp_sum
print(f"{dsl_avg or 0} {cpp_avg or 0} {dsl_inst} {cpp_inst}")
PY
        )

        # Parse Python-level wall-clock from bench stdout. The bench's summary
        # line looks like:
        #   <tag>   <MxN>   <dir>   C++_us   DSL_us   C++_GBps   DSL_GBps   DSL/C++
        # Match by shape + direction so it's robust to combo-tag variations.
        WALL=$(awk -v shape="${M}x${N}" -v dir="$DIR" '
            $2 == shape && $3 == dir {
                printf "%s %s", $5, $4   # DSL_us, C++_us
                exit
            }
        ' "${OUT}.stdout")
        WALL="${WALL:-0 0}"

        echo "${COMBO} ${M}x${N} ${DIR} ${PARSED} ${BYTES} ${CPP_BYTES} ${WALL}" >> "$RESULTS_FILE"
        GENERATED_FILES+=("${OUT}")
    done
done
done

# --- print summary ---
echo
# Section 1: kernel-only runtime (nsys cuda_gpu_kern_sum avg)
echo "[1/2] Kernel-only runtime (nsys cuda_gpu_kern_sum, WARMUP=${WARMUP} ITERS=${ITERS})"
echo "============================================================================================================="
printf "%-14s  %-14s  %-5s  %9s  %9s  %9s  %9s  %8s\n" \
    "combo" "shape" "dir" "DSL us" "C++ us" "DSL GB/s" "C++ GB/s" "DSL/C++"
printf -- "--------------  --------------  -----  ---------  ---------  ---------  ---------  --------\n"

while read -r COMBO SHAPE DIR DSL_NS CPP_NS DI CI BYTES CPP_BYTES DSL_WALL_US CPP_WALL_US; do
    DSL_US=$(awk -v v="$DSL_NS" 'BEGIN{printf "%.2f", v/1000.0}')
    CPP_US=$(awk -v v="$CPP_NS" 'BEGIN{printf "%.2f", v/1000.0}')
    DSL_GBPS=$(awk -v b="$BYTES" -v t="$DSL_NS" 'BEGIN{if(t>0) printf "%.1f", b/t; else print "n/a"}')
    CPP_GBPS=$(awk -v b="$CPP_BYTES" -v t="$CPP_NS" 'BEGIN{if(t>0) printf "%.1f", b/t; else print "n/a"}')
    if awk -v d="$DSL_NS" 'BEGIN{exit (d>0)?0:1}'; then
        SPEEDUP=$(awk -v d="$DSL_NS" -v c="$CPP_NS" 'BEGIN{printf "%.3fx", c/d}')
    else
        SPEEDUP="n/a"
    fi
    printf "%-14s  %-14s  %-5s  %9s  %9s  %9s  %9s  %8s\n" \
        "$COMBO" "$SHAPE" "$DIR" "$DSL_US" "$CPP_US" "$DSL_GBPS" "$CPP_GBPS" "$SPEEDUP"
done < "$RESULTS_FILE"

# Section 2: Python-level wall-clock from the bench (includes Python wrapper +
# kernel + sync; the L2-evict kernel itself is excluded by the bench).
echo
echo "[2/2] Python-level wall-clock (bench perf_counter_ns, includes wrapper overhead)"
echo "============================================================================================================="
printf "%-14s  %-14s  %-5s  %9s  %9s  %9s  %9s  %8s\n" \
    "combo" "shape" "dir" "DSL us" "C++ us" "DSL GB/s" "C++ GB/s" "DSL/C++"
printf -- "--------------  --------------  -----  ---------  ---------  ---------  ---------  --------\n"

while read -r COMBO SHAPE DIR DSL_NS CPP_NS DI CI BYTES CPP_BYTES DSL_WALL_US CPP_WALL_US; do
    # Wall-clock GB/s = bytes / (us * 1000)  [since us → ns multiply by 1000]
    DSL_WALL_GBPS=$(awk -v b="$BYTES" -v t="$DSL_WALL_US" 'BEGIN{if(t>0) printf "%.1f", b/(t*1000); else print "n/a"}')
    CPP_WALL_GBPS=$(awk -v b="$CPP_BYTES" -v t="$CPP_WALL_US" 'BEGIN{if(t>0) printf "%.1f", b/(t*1000); else print "n/a"}')
    if awk -v d="$DSL_WALL_US" 'BEGIN{exit (d>0)?0:1}'; then
        WALL_SPEEDUP=$(awk -v d="$DSL_WALL_US" -v c="$CPP_WALL_US" 'BEGIN{printf "%.3fx", c/d}')
    else
        WALL_SPEEDUP="n/a"
    fi
    printf "%-14s  %-14s  %-5s  %9s  %9s  %9s  %9s  %8s\n" \
        "$COMBO" "$SHAPE" "$DIR" "$DSL_WALL_US" "$CPP_WALL_US" \
        "$DSL_WALL_GBPS" "$CPP_WALL_GBPS" "$WALL_SPEEDUP"
done < "$RESULTS_FILE"

echo
echo "==> Generated files (${#GENERATED_FILES[@]} run(s)):"
for OUT in "${GENERATED_FILES[@]}"; do
    echo "    ${OUT}.nsys-rep"
    echo "    ${OUT}.sqlite"
    echo "    ${OUT}.stdout"
done
