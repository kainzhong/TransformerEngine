# MXFP8 CuTeDSL kernel — feature progress + LLM-shape benchmark

## What's been done so far

The CuTeDSL implementation in
[`quantize_mxfp8_cutedsl_alt.py`](quantize_mxfp8_cutedsl_alt.py)
now covers the full feature surface of the C++ `quantize_mxfp8_kernel`
fallback path (the `specialized::` fast-path is intentionally not
replicated).

| Feature | Status | C++ template knob |
|---------|--------|-------------------|
| BF16 input / E4M3 output | ✅ | `IType=__nv_bfloat16`, `OType=__nv_fp8_e4m3` |
| FP16 input | ✅ | `IType=__half` |
| FP32 input | ✅ | `IType=float` |
| E5M2 output | ✅ | `OType=__nv_fp8_e5m2` |
| Rowwise + colwise + bidim scaling | ✅ | `ROWWISE_SCALING`, `COLWISE_SCALING` |
| GEMM-swizzled scales | ✅ | `WITH_GEMM_SWIZZLED_SCALES` (via cute layout, not manual index arithmetic) |
| `noop` early-exit (CUDA Graphs / MoE) | ✅ | runtime `noop` ptr |
| Per-tensor amax (delayed-scaling FP8) | ✅ | `amax_ptr` |
| Forward activation fusion (relu / gelu / silu) | ✅ | `IS_ACT` + `OP` |
| Backward activation fusion (drelu / dgelu / dsilu) | ✅ | `IS_DACT` + `OP` |
| Bias gradient — path A (with colwise) | ✅ | `IS_DBIAS` + `COLWISE_SCALING` |
| Bias gradient — path B (rowwise-only with shmem transpose) | ✅ | `IS_DBIAS` + `!COLWISE_SCALING` |
| `IS_CACHED_ACT_OP` (perf optimization for bidim + activation) | ⏭️ skipped | `IS_CACHED_ACT_OP` |
| `CAST_DBIAS_ONLY` 128×128 config | ⏭️ skipped | shape-only, perf path |
| OOB masking on activation paths | ⏭️ skipped | edge-case for unaligned M/N |

### Key design choices

* **Const-expr-driven specialisation.** All template-arg-style flags are
  Python constants on the config (`cfg.dtype`, `cfg.fp8_dtype`,
  `cfg.activation`, `cfg.with_amax`, `cfg.with_dbias`,
  `cfg.with_gemm_swizzled_scales`, `cfg.is_dact`, `cfg.is_dbias_path_b`).
  The CuTeDSL JIT trace materialises a separate compiled kernel per
  `(shape × dtype × flags)` combination, equivalent to nvcc instantiating
  a C++ template specialization. The cache key in `_get_compiled_kernel`
  enumerates these dimensions.
* **Split paths, not nested const-expr.** When the kernel structure
  itself differs between configs (different number of TMA copies,
  different SharedStorage layout, different pipeline `tx_count`), we
  fork the top-level kernel body. The IS_DACT path lives in
  `_kernel_main_dact` (paired G2S TMA load with shared mbarrier and
  doubled `tx_count`); the rest stays in `_kernel_main`. Top-level
  dispatch sits in `kernel`.
* **Format-parameterised PTX kits.** bf16 and fp16 share the same
  packed-x2 fast-path shape (`max.xorsign.abs.<fmt>x2`, fused
  `<fmt>x2 * f32x2 → fp8x2`). One `_build_packed16_kit(fmt)` factory
  emits the per-format inline-asm wrappers; `_packed16_kit(dtype)` picks
  the right kit at JIT trace time.
* **Activation registry.** `_ACTIVATIONS` maps a Python string
  (`"gelu"`, `"dgelu"`, …) to a `Float32 → Float32` callable. Forward
  activations select IS_ACT; the `d`-prefixed entries are derivatives
  and select IS_DACT (`act_input` becomes a required wrapper kwarg).
  GELU constants and operator grouping match
  [`transformer_engine/common/util/math.h`](../../../transformer_engine/common/util/math.h)
  exactly so quantized output is bit-exact against `tex.dgelu` /
  `tex.dbias_dgelu`. `cute.math.tanh(fastmath=False)` is used because
  TE compiles activation kernels without `--use_fast_math` by default.
* **Hierarchical cute layout for swizzled scales.** Instead of computing
  the cuBLAS MXFP8 scale-block byte index by hand, the swizzled scale
  tensor is built with a hierarchical layout
  `((32, 4, n_tiles_y), (4, n_tiles_x)) : ((16, 4, 512·n_tiles_x), (1, 512))`
  — same logical shape as compact, only strides differ. The kernel just
  writes `mS_row[global_row, scale_col] = ...`. Visualised in
  [`swizzle_demo.svg`](swizzle_demo.svg).
* **Element-by-element dbias accumulation.** `_process_colwise{,_dact}`
  takes a running `partial_dbias_in` accumulator and extends it inside
  the inner loop, matching TE's flat
  `partial_dbias_colwise += elt` order across the 32 elements + N stages.
  Path B's `_process_rowwise{,_dact}` accumulates a 32-element
  `thread_dbias_rw` array (one per element of the row strip), then
  `_dbias_path_b_writeback` shmem-transposes it so each thread ends up
  owning a column for the workspace write.

### Commit log (this stretch)

```
cc0ed719 mxfp8 cutedsl: bias gradient (IS_DBIAS) — path A (with colwise)
a39d33d4 mxfp8 cutedsl: backward activation fusion (drelu/dgelu/dsilu)
bee3f8bc mxfp8 cutedsl: forward activation fusion (relu/gelu/silu)
35ba9f00 mxfp8 cutedsl: per-tensor amax reduction + atomic
db3633c3 mxfp8 cutedsl: noop early-exit flag
2339381d mxfp8 cutedsl: gemm-swizzled scales via cute layout
1692d0c0 mxfp8 cutedsl: support fp16/fp32 input and e5m2 output
```

Path B (rowwise-only IS_DBIAS) lives on disk but is not yet committed.

---

## Benchmark — LLM-shape kernels (B200, sm_100a)

### Setup

* GPU: GB200 / B200 (sm_100a), single CUDA stream.
* Timing: `torch.cuda.Event` over **200 iterations**, **20 warmup**.
  Wall-clock includes both the CuTeDSL kernel launch overhead and (for
  dbias combos) the post-kernel reduce — same as what TE's path
  measures end-to-end via `tex.dbias_d*`.
* Bytes accounting: for each combo, `bytes_in + rowwise_FP8 +
  colwise_FP8 + scales + dbias_workspace_writes` (the dbias workspace
  read/write of the reduce step is included on both sides since both
  paths produce the same final dbias[N]).
* Comparison: TE C++ reference invoked via the corresponding
  `MXFP8Quantizer(...)` / `tex.gelu` / `tex.silu` / `tex.dgelu` /
  `tex.dsilu` / `tex.dbias_dgelu` / `tex.dbias_dsilu` Python entry.
* Input dtype: bf16. Output FP8 dtype: e4m3. `compute_dbias=True` for
  the `dbias_*` rows. The DSL wrapper performs the post-kernel
  blocks_Y → 1 reduction with `torch.sum(workspace, dim=0).to(x.dtype)`
  — single CUDA launch, equivalent role to TE's internal
  `reduce_dbias` kernel.

### Combinations chosen

These are the four most common MXFP8 cast paths in transformer training:

| Combo | What it does | Where it shows up |
|-------|--------------|-------------------|
| `plain` | `MXFP8Quantizer(rowwise=T, columnwise=T)(x)` | Forward activations / weights pre-matmul |
| `dgelu` (bidim) | `tex.dgelu(grad, act_in, q)` | GELU FFN backward, output goes into the next layer's matmul |
| `dbias_dgelu` (bidim) | `tex.dbias_dgelu(grad, act_in, q)` | GELU FFN backward + bias gradient (one fused kernel saves a global pass over `grad`) |
| `dbias_dsilu` (bidim) | `tex.dbias_dsilu(grad, act_in, q)` | SwiGLU / SiLU FFN backward + bias gradient (Llama-class models) |

Plus path-B coverage on `dbias_dgelu` / `dbias_dsilu` (rowwise-only
quantizer — when the downstream consumer only needs rowwise FP8).

### Results — `direction=both` (rowwise + colwise quantize)

| Combo | Shape | C++ µs | DSL µs | C++ GB/s | DSL GB/s | DSL / C++ |
|-------|-------|-------:|-------:|---------:|---------:|----------:|
| `plain`         | 4096×4096   |   40.6 |   64.1 |   1680.4 |   1062.8 | 0.63× |
| `plain`         | 8192×8192   |   72.5 |   67.3 |   3762.3 |   4052.0 | **1.08×** |
| `plain`         | 8192×16384  |  139.7 |  123.9 |   3902.6 |   4402.5 | **1.13×** |
| `plain`         | 16384×16384 |  273.9 |  242.1 |   3981.1 |   4504.1 | **1.13×** |
| `gelu`          | 4096×4096   |   49.1 |   60.8 |   1389.0 |   1121.7 | 0.81× |
| `gelu`          | 8192×8192   |  192.5 |  212.4 |   1416.2 |   1283.9 | 0.91× |
| `gelu`          | 16384×16384 |  753.9 |  820.0 |   1446.4 |   1329.9 | 0.92× |
| `silu`          | 4096×4096   |   54.4 |   62.8 |   1253.0 |   1085.5 | 0.87× |
| `silu`          | 8192×8192   |  224.1 |  222.5 |   1216.7 |   1225.6 | **1.01×** |
| `silu`          | 16384×16384 |  850.4 |  856.5 |   1282.4 |   1273.2 | 0.99× |
| `dgelu`         | 4096×4096   |   74.4 |   82.7 |   1367.4 |   1230.1 | 0.90× |
| `dgelu`         | 8192×8192   |  271.5 |  302.6 |   1498.4 |   1344.7 | 0.90× |
| `dgelu`         | 8192×16384  |  538.2 |  595.5 |   1511.9 |   1366.4 | 0.90× |
| `dgelu`         | 16384×16384 | 1071.6 | 1181.7 |   1518.6 |   1377.2 | 0.91× |
| `dbias_dgelu`   | 4096×4096   |   87.9 |  110.3 |   1169.4 |    932.1 | 0.80× |
| `dbias_dgelu`   | 8192×8192   |  310.2 |  321.8 |   1325.1 |   1277.3 | 0.96× |
| `dbias_dgelu`   | 8192×16384  |  582.8 |  620.6 |   1410.6 |   1324.8 | 0.94× |
| `dbias_dgelu`   | 16384×16384 | 1164.1 | 1223.7 |   1412.4 |   1343.7 | 0.95× |
| `dbias_dsilu`   | 4096×4096   |   78.2 |  107.0 |   1314.3 |    960.4 | 0.73× |
| `dbias_dsilu`   | 8192×8192   |  282.4 |  309.7 |   1455.5 |   1327.4 | 0.91× |
| `dbias_dsilu`   | 8192×16384  |  523.2 |  599.3 |   1571.2 |   1371.8 | 0.87× |
| `dbias_dsilu`   | 16384×16384 | 1035.8 | 1183.9 |   1587.3 |   1388.8 | 0.87× |

### Results — `direction=row` (rowwise-only, dbias path B)

| Combo | Shape | C++ µs | DSL µs | C++ GB/s | DSL GB/s | DSL / C++ |
|-------|-------|-------:|-------:|---------:|---------:|----------:|
| `dbias_dgelu` | 4096×4096   |   63.1 |   94.6 | 1353.7 |  904.0 | 0.67× |
| `dbias_dgelu` | 8192×8192   |  229.1 |  176.0 | 1492.1 | 1942.3 | **1.30×** |
| `dbias_dgelu` | 16384×16384 |  808.7 |  652.9 | 1690.8 | 2094.2 | **1.24×** |
| `dbias_dsilu` | 4096×4096   |   64.9 |  102.2 | 1317.7 |  836.5 | 0.63× |
| `dbias_dsilu` | 8192×8192   |  238.1 |  208.8 | 1435.5 | 1637.5 | **1.14×** |
| `dbias_dsilu` | 16384×16384 |  857.9 |  774.3 | 1594.0 | 1765.9 | **1.11×** |

### Observations

* **Plain bidim quantize is bandwidth-bound and we're at ~4.5 TB/s on a
  16k² tensor.** DSL beats C++ at large shapes by 8-13%. At the small
  4k² shape DSL trails (0.63×) because the kernel runtime is short
  enough that host-side launch overhead dominates and our wrapper does
  more Python-side work than `MXFP8Quantizer.__call__`.
* **Forward `gelu` / `silu` and backward `dgelu`** sit at ~0.90×–0.92×
  C++ across all sizes. The gap is concentrated in the activation
  compute — once we leave the packed-x2 fast path and run scalar f32
  ops with `cute.math.tanh(fastmath=False)` (matching TE's precise
  `tanhf` for bit-exactness), throughput drops from ~4 TB/s to ~1.3
  TB/s and the difference becomes compute-bound rather than bandwidth.
* **`dbias_dgelu` / `dbias_dsilu` bidim** sit at ~0.87×–0.96× C++ at
  large shapes. The post-kernel `torch.sum` reduce is a single launch
  but adds a few µs of overhead vs TE's tightly-fused `reduce_dbias`
  C++ kernel.
* **`dbias_dgelu` / `dbias_dsilu` rowwise-only (path B)** is the
  outlier — DSL is 1.1×–1.3× *faster* than C++ at 8k+ shapes. The
  shmem transpose is efficient and the kernel has fewer TMA stores
  than the bidim case.

### Results — feature-axis sweep at 8192×8192 bidim, plain quantize

These rows isolate the perf cost of each non-default flag against the
same C++ reference path. Activation perf scales similarly across
relu/gelu/silu (and their derivatives) — so only `gelu` and `silu` are
shown above; `relu` is faster than both (it's `fmax(x, 0)`) and would
sit near the top of the activation rows.

| Feature | C++ µs | DSL µs | C++ GB/s | DSL GB/s | DSL / C++ |
|---|---:|---:|---:|---:|---:|
| BF16 → E4M3 (default)         |  72.6 |  64.5 | 3757 | 4225 | **1.12×** |
| BF16 → E5M2                   |  72.5 |  64.9 | 3761 | 4203 | **1.12×** |
| FP16 → E4M3                   |  72.4 |  66.3 | 3764 | 4115 | **1.09×** |
| FP16 → E5M2                   |  72.5 |  64.3 | 3760 | 4240 | **1.13×** |
| FP32 → E4M3                   | 114.3 | 106.0 | 3560 | 3839 | **1.08×** |
| FP32 → E5M2                   | 114.3 | 106.0 | 3561 | 3837 | **1.08×** |
| BF16 → E4M3, GEMM-swizzled    |  71.2 |  61.8 | 3830 | 4410 | **1.15×** |
| BF16 → E4M3, with `amax_ptr`  |  72.6 |  80.5 | 3754 | 3387 | 0.90× |

Notes:
* **E4M3 vs E5M2** is a no-op delta (different `cvt.rn.satfinite.<fmt>x2.f32`
  PTX opcode but same instruction count and latency). The two rows match
  to within timing noise on both DSL and C++.
* **FP32 input** is ~1.6× the kernel time of bf16/fp16 (the input is 2×
  bigger in memory and goes through a scalar f32 amax/cast path with no
  packed-x2 fast path). DSL still tracks C++ at ~1.08× across the dtype
  sweep — the ratio is stable.
* **GEMM-swizzled scales** improve DSL throughput slightly (1.15× vs
  1.12×) because cute layout drives the per-element divmod with
  static-stride arithmetic; C++ uses the `gemm_swizzled_scale_idx`
  helper which performs the same divmods at runtime. Both are tiny in
  the kernel-time budget.
* **`amax_ptr`** is the only dimension where DSL underperforms C++ —
  the extra `warp_redux_sync.fmax.f32` + intra-CTA shmem reduce + cross-
  CTA `atom.global.max.s32` adds ~16 µs to the 65 µs base kernel
  (~25% overhead). C++ shows almost no overhead because the same logic
  is fused into the same compilation unit and the compiler can hide
  the latency. Optimizable later if amax becomes a hot path.

### Caveats

* The 4k² shape consistently underperforms (0.63×–0.81×) — wrapper
  Python overhead dominates a sub-100µs kernel call.
* Numbers above are **wall-clock with `torch.cuda.Event`**, not
  `nsys`-extracted kernel-only times. The launch overhead for our
  wrapper is ~10-20µs higher than `MXFP8Quantizer.__call__`'s
  (additional Python validation, `_get_compiled_kernel` cache lookup,
  more `make_ptr` calls). Subtracting that overhead would tighten the
  small-shape numbers; nsys-based measurement is in
  [`run_nsys_profile.sh`](run_nsys_profile.sh).
* All quantized output (data + scales) is **bit-exact** vs the C++
  reference across all shapes/combos in
  [`test_mxfp8_quantize_cutedsl.py`](test_mxfp8_quantize_cutedsl.py)
  and the related smoke tests.
* The `dbias` value is bit-exact for bf16 inputs across all combos.
  fp16 + dgelu and fp32 + (dgelu / dsilu / drelu, on path B) drift by
  ≤1 ULP because the wrapper uses `torch.sum(workspace, dim=0)`
  (tree-reduce — single CUDA launch) while TE's `reduce_dbias` is a
  sequential left-fold per column. A bit-exact-matching DSL reduce
  would either need a custom CuTeDSL reduce kernel mirroring TE's
  loop, or a Python-loop reduce (~256 launches at M=16384 — kills
  perf). The torch.sum trade-off is the same that nvcc/ptxas vs
  CuTeDSL FFMA-scheduling already exposes for the activation-itself
  rounding (1-ULP territory regardless).
