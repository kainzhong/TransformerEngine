# MXFP8 cast: specialized fast-path vs. general fallback (C++ kernels)

Reference notes on the two C++ MXFP8 quantize kernels in TE — the
`specialized::` cast-only fast path and the general
`quantize_mxfp8_kernel` fallback. Useful when reading the perf charts
or when deciding what the CuTeDSL kernel is actually being compared
against.

- **Specialized**: [`transformer_engine/common/cast/mxfp8/specialized/quantize_mxfp8.cuh`](../../../transformer_engine/common/cast/mxfp8/specialized/quantize_mxfp8.cuh)
- **Fallback**:    [`transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh`](../../../transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh)

## TL;DR

Same per-thread chunk granularity (1 thread per 32-element scale
block, in both kernels). What differs is the **CTA tile shape**, the
**smem-staging strategy**, the **PTX instructions used for cast and
amax**, and the **feature surface** they support.

The specialized kernel is **plain cast only** (no dbias / no
activation / fp16 or bf16 inputs only) and **opt-in via env var**
(`ENABLE_CAST_ONLY=1`). Everything else routes to the fallback.

## When does the specialized kernel run?

```cpp
// quantize_mxfp8_spec.cuh:106 — default
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename IType, typename OType>
inline bool hasSpec() { return false; }

// only these specializations override to true:
template <> inline bool hasSpec<false, false, false, fp16, fp8e5m2>() { return is_cast_only_enabled(); }
template <> inline bool hasSpec<false, false, false, fp16, fp8e4m3>() { return is_cast_only_enabled(); }
template <> inline bool hasSpec<false, false, false, bf16, fp8e5m2>() { return is_cast_only_enabled(); }
template <> inline bool hasSpec<false, false, false, bf16, fp8e4m3>() { return is_cast_only_enabled(); }
```

So the kernel is selected only when **all** of the following hold:

1. `IS_DBIAS=false`, `IS_DACT=false`, `IS_ACT=false` (plain cast — no fused ops)
2. `IType ∈ {fp16, bf16}`, `OType ∈ {fp8e4m3, fp8e5m2}` (no fp32 input)
3. `ENABLE_CAST_ONLY=1` is set in the environment

Anything outside this matrix — fp32 input, any fused op flag, env var
unset — falls back to `quantize_mxfp8_kernel` in the parent file.

## Three specialized kernels, not one

The specialized header actually defines **three separate kernels**,
selected by `(rowwise, colwise)` `CastTraits` specialization:

| Kernel | `(rowwise, colwise)` | Strategy |
|--|--|--|
| Rowwise-only | `(true, false)` | gmem → registers direct, no input smem stage |
| Colwise-only | `(false, true)` | gmem → registers direct, no input smem stage |
| Bidim       | `(true, true)`  | gmem → smem (TMA + mbarrier double-buffer), like the fallback |

The unidirectional kernels and the bidim kernel are structurally
different — only the bidim one needs smem staging.

## Unidirectional kernels (rowwise-only, colwise-only)

These are the kernels people usually mean when they say "the
specialized fast path."

| Aspect | Fallback (rowwise) | Specialized rowwise-only |
|--|--|--|
| `CHUNK_DIM_Y × CHUNK_DIM_X` (CTA tile) | 64 × 64 (= 4096 elts) | 4 × 1024 (= 4096 elts) |
| Threads per CTA | 64 (32 × 2 thread grid) | 128 (4 warps × 32) |
| Per-thread payload | 1 row × 32 cols (one scale block) | 1 row × 32 cols (one scale block) |
| Input staging | gmem → smem (TMA, `STAGES=2`) → registers | gmem → registers directly (`uint4` vector load, `STAGES=1`) |
| Scale staging | written direct to gmem from registers | staged in smem first (`_cache_rowwise_scale_in_smem=true`), then one coalesced shmem→gmem write |
| Tile shape rationale | Square, supports a colwise reread of the same staged data | Long thin strip — maximizes coalesced `uint4` loads per warp; no reread |

Per-thread granularity is **identical** — each thread owns exactly
one 32-element scale block in both kernels. The amax is computed
serially in registers, no warp shuffles. The win for the
specialized unidirectional path comes from skipping the smem-stage
round-trip (which the fallback only does to enable bidim's reuse and
isn't needed when there's only one direction), plus the wider PTX
instructions described below.

## Bidim specialized kernel — looks much more like the fallback

`CastTraits<IType, OType, true, true>` (line 579 of the spec header)
defines the bidim variant, and structurally it's close to the
fallback:

| Aspect | Fallback bidim | Specialized bidim |
|--|--|--|
| Input load | TMA (`cp_async_bulk_tensor_2d_global_to_shared`) | TMA (`cp_async_bulk_tensor_2d_global_to_shared`) — line 778 |
| Output store | TMA (`cp_async_bulk_tensor_2d_shared_to_global`) | TMA (`cp_async_bulk_tensor_2d_shared_to_global`) — lines 810/814 |
| Smem staging | input + output buffers, `STAGES=2` | input + output buffers, `numStages=2` |
| TMA swizzle | 128B input / 64B output | 128B input / 64B output (`_tma_swizzle=true`) |
| Producer/consumer mbarriers | yes | yes (`ldg_producer/consumer`, `stg_producer/consumer`) |
| Per-CTA warp count | 2 warps | 2 warps (`warpLayout = (1, 2)`) |
| Thread layouts (row vs col passes) | rotated 32×1 ↔ 1×32 | rotated 32×1 ↔ 1×32 (`rowThreadLayout` / `colThreadLayout`) |

So in bidim, both kernels look like: TMA-load → smem → 2-pass
(rowwise + colwise) compute in registers → TMA-store. Same
pipeline, same smem-staging story.

What the **bidim specialized** still wins on:

1. **The inner-loop PTX** — `cvt.4x` for casting and
   `fma_f32_bf16` for amax → e8m0 (see next section). The fallback's
   inner loop uses narrower variants because it has to support more
   `(IType, IS_*, …)` combinations.
2. **No template-arg overhead** — the specialized kernel has no
   `if constexpr (IS_DBIAS)` / `if constexpr (IS_ACT)` branches, no
   conditional accumulators in the loop body, no `IS_DACT` second
   input. The compiler emits a tighter instruction sequence.
3. **Hand-tuned warp/iter layout** — `iterLayout = Layout<1, 4>` and
   `warpLayout = Layout<1, 2>` give a wider blockDim than the
   fallback's square 64×64, which interacts well with the wider
   `cvt.4x` register footprint.

The smem and TMA usage themselves aren't the differentiators in
bidim — both kernels need them for the same reasons.

## PTX instructions for the inner loop

### amax → e8m0

```cpp
// Specialized — bf16 input (quantize_mxfp8_spec.cuh:73)
ptx::fma_f32_bf16(amax_fp32, reinterpret_cast<uint16_t&>(amax), max_norm_rcp);
return ptx::float_to_e8m0(amax_fp32);
```

Single instruction: takes a bf16 `amax` and a 16-bit `max_norm_rcp`
constant (precomputed per `(IType, OType)` pair, lines 35–62 of the
specialized header), produces `amax * (1/max_norm)` directly in fp32
without an explicit widen step. Then `float_to_e8m0` extracts the
exponent.

```cpp
// Fallback — go through fp32 explicitly
amax_fp32 = static_cast<float>(amax);                 // bf16→fp32 widen
amax_fp32 *= Quantized_Limits<OType>::max_norm_rcp;   // separate fp32 multiply
e8m0 = ptx::float_to_e8m0(amax_fp32);
```

### Cast fp32/bf16 → fp8

| | Fallback | Specialized |
|--|--|--|
| PTX op | `cvt.rn.satfinite.<fmt>x2.f32` (or scalar) | `cvt.rn.satfinite.<fmt>x4.bf16x4` / `f16x4` |
| Elements per instruction | 2 | 4 |
| Issues per 32-elt chunk | ~16 | ~8 |
| Why specialized can use the wider one | Whole 32-elt chunk lives in this thread's registers as a contiguous 16-reg group, so 4-elt cvt operands are naturally adjacent | Fallback's per-thread layout pairs match `cvt.x2` better; switching to 4x would require register repacking that costs more than it saves |

Together the wider cast + the bf16-operand FMA cut ~30–40% of the
per-element instruction count on the fast path.

## Feature surface

What each kernel supports:

| Feature | Fallback | Specialized |
|--|--|--|
| Plain cast bf16/fp16 → fp8e4m3/e5m2 | ✓ | ✓ (the only thing it does) |
| fp32 input | ✓ | ✗ — falls back |
| Rowwise scaling | ✓ | ✓ (dedicated rowwise-only kernel, no smem stage) |
| Colwise scaling | ✓ | ✓ (dedicated colwise-only kernel, no smem stage) |
| Bidim (rowwise + colwise in one pass) | ✓ | ✓ (dedicated bidim kernel — uses TMA+smem+mbarriers like the fallback) |
| `IS_ACT` (forward act fusion) | ✓ | ✗ |
| `IS_DACT` (backward act fusion) | ✓ | ✗ |
| `IS_DBIAS` (bias gradient) | ✓ | ✗ |
| `WITH_GEMM_SWIZZLED_SCALES` | ✓ | (not exposed in the spec header — would need separate spec) |
| `noop` early-exit | ✓ | (not in the cast-only kernel) |
| Per-tensor amax | ✓ | (not in the cast-only kernel) |

Everything outside the "plain cast bf16/fp16" cell falls back to the
general kernel. The specialized kernel exists *purely* to take the
hot path (which in inference workloads is by far the most common
op — plain weight/activation cast with no fused work) and run it on
the most efficient PTX combinations Blackwell offers.

## Why "1 thread per chunk" is faster than "32 threads cooperating"

This is a common misconception so worth stating explicitly: **neither
kernel uses 32 cooperating threads per scale block**. Both assign one
thread per 32-element block. The amax is computed in registers
serially, no warp shuffles involved.

If a hypothetical kernel *did* split the 32 elements across 32
threads, it would pay:
- a `__shfl_xor_sync` warp reduction for the amax (~6–10 cycles in
  the critical path);
- inability to use `cvt.4x` (which needs 4 inputs in one thread's
  registers);
- inability to use `fma_f32_bf16` per-block (the per-element widening
  would need shuffles to assemble);
- worse ILP — 32 threads doing the same step in lockstep cannot
  interleave loads/computes/stores across chunks the way 32 threads
  on independent chunks can.

The win is from running 32 chunks worth of independent work per warp
(maximum ILP) rather than 32 threads cooperating on 1 chunk.

## Implications for benchmarking

The CuTeDSL kernel in this repo replicates the **fallback's** feature
surface (bidim, IS_ACT, IS_DACT, IS_DBIAS, swizzled scales, amax,
noop). Its bench compares against the fallback kernel by default —
this is the apples-to-apples target.

If you want to compare against the specialized fast path:

```bash
ENABLE_CAST_ONLY=1 ./run_nsys_profile.sh \
    --shapes '4096,4096;8192,8192;16384,16384' \
    --direction row,col,both \
    --combos plain
```

This only changes the `plain` combo's reference (across all three
directions, since the specialized kernel covers rowwise / colwise /
bidim each with its own variant). All other combos still hit the
fallback regardless of the env var, because the specialized header
only registers `<IS_DBIAS=false, IS_DACT=false, IS_ACT=false>`.

The current charts in `perf_charts.svg` use the fallback path for all
combos, including `plain`. If you ran the specialized path on the
plain combo, expect the C++ side to gain ~30–50% throughput on bf16/fp16
plain cast at large shapes — narrowing or inverting the DSL/C++ ratio
for that one bar.
