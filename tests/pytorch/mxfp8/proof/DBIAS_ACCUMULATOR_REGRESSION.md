# Colwise dbias accumulator: why carrying it across tiles costs ~1.3%

**Commit under test:** `6090a719` ("tmp") vs parent `fd9b1d82`, branch `cutedsl_mxfp8_common`
**Hardware:** NVIDIA GB200 (CC 10.0), 129.2 MB L2
**Date:** 2026-08-05

---

## Verdict

The commit is a **pure refactor with no measured upside and a reproducible ~1.2-1.4%
regression** on the bandwidth-bound activations.

| config | effect |
|---|---|
| `dbias`, `dbias_dgelu`, `dbias_dsilu`, `dbias_dqgelu` | neutral (within noise) |
| `dbias_drelu`, `dbias_dsrelu` (both + swizzle-on) | **+1.2% to +1.4% slower** |

It executes *fewer* instructions (-0.40%) and still takes longer. The cause is a
loop-carried dependency that reduces the compiler's instruction-scheduling freedom,
which at this kernel's low occupancy converts directly into lost memory-level
parallelism.

---

## What the commit changes

Semantically identical. Both versions sum the same values in the same order; only the
*shape of the dataflow* differs.

**Before** — the accumulator is reset per tile, and tiles are joined by an outer add:

```python
# outer, once per CTA
block_dbias = Float32(0.0)

for tile_idx in cute.range(num_tiles, unroll=1):
    ...
    amax_c, dbias_c = quantize_colwise_mxfp8(...)   # dbias_partial = 0.0 inside
    block_dbias += dbias_c                          # one add joins tile N to N-1
```

**After** — the accumulator is threaded through the call:

```python
# outer, once per CTA
block_dbias = Float32(0.0)

for tile_idx in cute.range(num_tiles, unroll=1):
    ...
    amax_c, dbias_c = quantize_colwise_mxfp8(..., block_dbias)  # dbias_partial = block_dbias
    block_dbias = dbias_c                                       # just a move
```

The commit message frames this as parity with the CUDA C++ kernel. It is **not** a
correctness fix — the old form already accumulated across tiles via the outer `+=`.

It also dropped the `if DBIAS_REDUCTION_COLWISE:` guard, so `block_dbias` is now
unconditionally live for *every* config, including `plain`/`gelu` which never read it.

---

## Measurement

`run_mxfp8_benchmark.py` (copied from `cutedsl_mxfp8_common_bench`), GPU mode: kernel
time from nsys NVTX range kernel summary, cold L2, bf16 in / e4m3 out.

Because the commit touches only a pure-Python CuTeDSL file loaded from the source tree,
and the editable install (`2.19.0.dev0+fd9b1d82`) is built from the *parent* commit, both
sides of the A/B share a byte-identical `libtransformer_engine.so`. No rebuild is
involved, so the C++ backend column acts as a **built-in noise control**: it is the same
code in both runs, and any movement there is measurement noise.

L2 eviction is unconditional in this path (`bench_mxfp8_cutedsl.py`), not behind
`--evict-l2`, which only governs the non-nsys path:

```python
for i in range(iters):
    evict.zero_()          # 256 MB buffer vs 129.2 MB L2, OUTSIDE the NVTX range
    torch.cuda.synchronize()
    nvtx.range_push(rng)
    fn()                   # only this is measured
    torch.cuda.synchronize()
    nvtx.range_pop()
```

`run_mxfp8_benchmark.py` additionally filters `memset`/`fill`/`elementwise` kernels out
of the nsys attribution (`_EVICT_NAME_PATTERNS`).

### Broad sweep — 5 activations x 2 directions x 2 swizzle x 6 shapes

| activation | n | DSL mean | slower in | sign-test p | CPP control slower in |
|---|---|---|---|---|---|
| `dbias_dgelu` | 24 | -0.034% | 10/24 | 1.00 | 11/24 |
| `dbias_dqgelu` | 24 | -0.192% | 8/24 | 1.00 | 1/24 |
| `dbias_drelu` | 24 | +0.163% | 12/24 | 1.00 | 7/24 |
| `dbias_dsilu` | 24 | +0.090% | 12/24 | 1.00 | 9/24 |
| `dbias_dsrelu` | 24 | +0.588% | **22/24** | **<0.0001** | 10/24 |

> **Caveat on `drelu` in this table.** The 12/24 reading is an artifact. `drelu` was
> measured in a noisy window — its C++ control had sd 0.68 and a +-2.31% range, enough to
> bury a ~1.2% signal. The isolated re-test below shows it regresses just like `dsrelu`.
> There is no `drelu`/`dsrelu` asymmetry.

### Isolated re-test — `both` + swizzle-on, `--iters 300`, alternating A/B/A/B

| shape | `dsrelu` r1 | `dsrelu` r2 | `drelu` r1 | `drelu` r2 |
|---|---|---|---|---|
| 2048x5120 | +1.20% | +0.96% | +1.06% | +1.06% |
| 4096x4096 | +1.12% | +1.02% | +0.68% | +0.84% |
| 4096x8192 | +1.31% | +1.31% | +1.19% | +1.25% |
| 8192x8192 | +1.44% | +1.41% | +1.31% | +1.40% |
| 4096x14336 | +1.68% | +1.69% | +1.42% | +1.36% |
| 8192x28672 | +1.82% | +2.04% | +1.85% | +1.73% |
| **mean** | **+1.43%** | **+1.41%** | **+1.25%** | **+1.27%** |

Both: **12/12 slower**, rep-to-rep agreement within 0.25pp, C++ control flat
(-0.02% / +0.08% for `dsrelu`, -0.00% / -0.01% for `drelu`). The effect scales
monotonically with problem size, consistent with a **per-tile** cost.

---

## ncu evidence

`dbias_dsrelu`, 4096x14336, `both`, swizzle-on, steady-state launch
(`--launch-skip 1 --launch-count 1`).

Everything structural is identical — this rules out occupancy, register pressure, and
any cache/footprint explanation:

| | HEAD | BASE |
|---|---|---|
| registers/thread | 64 | 64 |
| occupancy (% peak warps) | 26.38 | 26.35 |
| shared mem/block (dynamic) | 24832 B | 24832 B |
| grid size / waves per SM | 14336 / 10.48 | 14336 / 10.48 |
| DRAM bytes read / written | 234.91 MB / 91.94 MB | 234.95 MB / 91.94 MB |

The actual difference:

| metric | HEAD | BASE | delta |
|---|---|---|---|
| **instructions executed** | 50.12 M | 50.32 M | **-0.40%** |
| **time** | 62208 ns | 61056 ns | **+1.89%** |
| DRAM throughput (% peak) | 66.31 | 67.61 | -1.92% |
| issue active (% peak) | 74.73 | 77.15 | -3.14% |
| **stall: `wait`** | 0.71 | 0.61 | **+16.4%** |
| stall: `short_scoreboard` | 0.51 | 0.40 | +27.5% |
| stall: `long_scoreboard` | 1.46 | 1.42 | +2.8% |

The ncu-measured +1.89% matches the nsys-measured +1.68%/+1.69% for this shape.

**The commit executes fewer instructions and still takes longer.** The refactor did
remove real work; the kernel simply is not instruction-count limited.

---

## Mechanism

### What `stall: wait` is

Full metric name: `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`

- **`smsp`** — SM sub-partition. Each SM has 4, each with its own warp scheduler.
- **`average_warps_issue_stalled_wait`** — average number of resident warps sitting in
  the `wait` stall state.
- **`per_issue_active`** — normalized per cycle in which that scheduler *did* issue an
  instruction.

Read it as: *"on a cycle when this scheduler was doing useful work, how many warps were
blocked on `wait`?"* 0.61 -> 0.71 means about **0.1 more warps per sub-partition** parked
in that state at any moment.

The underlying model: each cycle a scheduler picks one **eligible** warp (operands ready,
required pipe free) and issues one instruction. Every non-eligible warp is *stalled*, and
ncu attributes a reason:

| stall reason | warp is waiting on... |
|---|---|
| **`wait`** | a **fixed-latency** result — plain arithmetic (FADD/FMUL/FFMA), ~4-6 cycles, latency known at compile time |
| `short_scoreboard` | a short variable-latency op — shared memory, MIO |
| `long_scoreboard` | a **global memory** load — hundreds of cycles |
| `not_selected` | nothing; it was ready, another warp just got picked first |

`wait` is the relevant one because it is specifically the signature of **back-to-back
dependent arithmetic**. On NVIDIA hardware the compiler knows ALU latencies exactly, so
it encodes stall cycles directly into the instruction stream — "issue this, then sit out
4 cycles." `wait` is warps serving those sentences.

### Why the change raises it

The compiler's job is to fill those mandatory gaps with *independent* instructions.
Whether it can depends on what is available to reorder.

**Before** — each tile's accumulator starts from a constant:

```
dbias_partial = 0.0          # depends on nothing
  ... 32 dependent adds ...  # chain starts fresh this tile
block_dbias += dbias_c       # one add, joins tiles at the end
```

Tile N's add chain is **independent of tile N-1's**. While waiting out a 4-cycle gap in
the accumulator chain, the scheduler has unrelated work to slot in — address math, the
srelu derivative, scale computation, stores. Gaps get filled; `wait` stays low.

**After** — the accumulator threads through:

```
dbias_partial = block_dbias  # depends on tile N-1's LAST add
  ... 32 dependent adds ...  # chain continues, unbroken
block_dbias = dbias_c        # just a move
```

A single dependency chain now runs the entire length of the tile loop. Fewer instructions
overall, but less freedom to reorder around them, so more fixed-latency gaps go unfilled.
`wait` rises.

### Why that costs wall-clock

Normally it would not. That is the GPU bargain: when one warp stalls, the scheduler
switches to another, and with enough resident warps the stall is invisible.

The catch is **occupancy is 26% of peak warps**. There often is no other eligible warp to
switch to, so the stall lands on the clock instead of being absorbed. This is why
instruction scheduling matters here at all — it matters exactly when occupancy is too low
to paper over it.

It surfaces as a *bandwidth* loss rather than a compute loss because those same warps
issue the loads for the next tile. Delay the warp -> delay the load issue -> fewer memory
requests in flight -> memory pipe runs less full. ncu shows the chain end to end: issue
rate -3.14%, achieved DRAM throughput -1.92%, on byte-for-byte identical traffic.

### Why only the cheap activations regress

The split is not by activation identity, it is by **proximity to the memory roofline**:

| activation | GB/s (both, sw-on) | regression |
|---|---|---|
| `drelu` / `dsrelu` | 4700-5800 | **~1.2-1.4%** |
| `dgelu` | ~3180 | none |
| `dsilu` / `dqgelu` | ~2200 | none |

`gelu`/`silu`/`qgelu` are compute-bound with slack to spare, so the added stall is
absorbed and measures as zero. `drelu`/`dsrelu` run close enough to the roofline that it
converts straight into time.

---

## Caveats

- **Inferred, not proven:** the stall/issue/bandwidth correlates are confirmed, but SASS
  was not inspected to watch the TMA load-issue point actually slip relative to the
  accumulator chain. The evidence is strongly consistent with that, not conclusive.
- **Coverage:** bf16 in / e4m3 out only. `quantize_colwise_mxfp8` branches on
  `USE_HALF_PRECISION = is_packed16(DTYPE) and ACTIVATION is None`, so fp32 takes a
  different route through the changed function. Untested.
- **Non-dbias configs untested.** Dropping the `DBIAS_REDUCTION_COLWISE` guard made
  `block_dbias` unconditionally live for `plain`/`gelu`/`silu` etc. Best case the compiler
  eliminates it; there is no mechanism by which it makes them faster.
- **Fragile effect.** A ~1.3% delta from a semantically-null change is a property of the
  current CuTeDSL/NVVM scheduler, not of the algorithm. It could vanish or invert on a
  toolchain bump. Decide this commit on code-clarity grounds, not on the number.

---

## Unrelated finding worth more than this commit

`dbias_dqgelu` and `dbias_dsilu` run at **0.82-1.02x** vs the CUDA C++ kernel — the
CuTeDSL backend is *slower* by up to 18%, on both HEAD and base. That is a pre-existing
gap an order of magnitude larger than anything in this diff.

---

## Reproduction

```bash
# A/B by swapping the one changed file; no rebuild needed.
Q=transformer_engine/common/CuTeDSL/cast/mxfp8/quantize_mxfp8.py
git show HEAD:$Q   > /tmp/q_HEAD.py
git show HEAD~1:$Q > /tmp/q_BASE.py

for side in HEAD BASE; do
  cp /tmp/q_$side.py $Q
  find . -name __pycache__ -path '*CuTeDSL*' -exec rm -rf {} +
  python tests/pytorch/mxfp8/run_mxfp8_benchmark.py --combos dbias_dsrelu \
    --directions both --swizzle on --modes gpu --iters 300 \
    --shapes "2048,5120;4096,4096;4096,8192;8192,8192;4096,14336;8192,28672" \
    > /tmp/out_$side.txt
done
git checkout -- $Q
```

ncu (note `--resolve-symbols=false` is required for **nsys** on this box, or
finalization hangs):

```bash
NVTE_ENABLE_CUTEDSL_QUANT_BACKEND=1 PYTHONPATH=$PWD ncu --csv \
  --kernel-name "regex:kernel_cutlass_kernel.*MXFP8QuantizeKernel" \
  --launch-skip 1 --launch-count 1 \
  --metrics launch__registers_per_thread,gpu__time_duration.sum,\
smsp__inst_executed.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio \
  python tests/pytorch/mxfp8/bench_mxfp8_cutedsl.py --gpu-nsys --combos dbias_dsrelu \
  --directions both --swizzles on --shapes "4096,14336" --in-dtypes bf16 --fp8s e4m3 \
  --warmup 1 --iters 1
```
