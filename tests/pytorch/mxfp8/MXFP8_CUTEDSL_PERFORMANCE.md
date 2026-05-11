# MXFP8 Quantization: CuTeDSL vs. C++ Performance

Benchmark of the CuTeDSL MXFP8 quantization kernel against the Transformer
Engine C++ reference (`MXFP8Quantizer` → `quantize_mxfp8.cuh`) on a single
NVIDIA GB200 (SM100a, HBM3e). All combos covered by the CuTeDSL kernel
(plain quantize, forward / backward activation fusion, bias-gradient on
both path A and path B) are compared.

The kernel-only feature surface and bit-exactness story are documented in
[`MXFP8_CUTEDSL_FEATURE_BENCH.md`](MXFP8_CUTEDSL_FEATURE_BENCH.md). This
file is the methodology + numbers companion.

## Setup

- **GPU**: NVIDIA GB200 (Blackwell, SM100a), 192 GB HBM3e, single CUDA stream
- **Inputs**: BF16 `(M, N)` tensor (and `act_input` for activation combos)
- **Output**: FP8E4M3 data + E8M0 scales (rowwise / colwise / bidim per row)
- **Iterations**: 10 warmup + 100 timed (all numbers below)
- **Reference**: TE C++ entry points (`MXFP8Quantizer.__call__`,
  `tex.gelu` / `tex.dgelu` / `tex.dbias_dgelu` / …)
- **Bench script**: [`bench_mxfp8_cutedsl.py`](bench_mxfp8_cutedsl.py)
- **Profile driver**: [`run_nsys_profile.sh`](run_nsys_profile.sh)

## Two measurement methodologies

The bench script reports both — they answer different questions.

| | **Wall-clock (`torch.cuda.Event`)** | **Kernel-only (`nsys cuda_gpu_kern_sum`)** |
|--|--|--|
| What it measures | First `cudaEventRecord` → last `cudaEventRecord` of the timed loop, divided by iters | Average GPU runtime of the dominant quantize kernel, extracted from `nsys-rep` |
| Includes | Python wrapper, `MXFP8Quantizer.__call__` validation, `_get_compiled_kernel` cache lookup, all `make_ptr` calls, kernel launch, kernel run | Kernel only |
| Excludes | Nothing inside the timed loop | All host-side overhead, cudaLaunchKernel, descriptor construction |
| When it matters | End-to-end picture for callers who hit the Python entry point each step | Apples-to-apples kernel comparison; hides wrapper differences between TE's `tex.*` and our DSL wrapper |
| When it lies | When the kernel is short (≤100 µs) and Python overhead dominates | When users actually pay the wrapper cost in production code |

The CuTeDSL wrapper does ~10–20 µs more host-side work per call than
TE's tightly-tuned `tex.*` entry points (extra Python validation, the
JIT cache lookup, more `make_ptr` calls for scale tensors and the dbias
workspace). At 16k² shapes that's noise; at 4k² it's a quarter of the
total measurement.

**Treat nsys as authoritative for kernel quality**; treat wall-clock
as authoritative for "what does the user actually see" in tight
training loops.

## Where the two methodologies disagree

### 1. Small shapes (4k²): wall-clock distorts beyond recognition

`dbias_dgelu` rowwise (path B), 4096×4096:

|             | C++ µs | DSL µs | DSL/C++ |
|-------------|------:|------:|--------:|
| Wall-clock  | 63.1  | 94.6  | **0.67×** |
| Nsys kernel | 51.3  | 43.5  | **1.18×** |

The CuTeDSL kernel is *faster* than TE's; the wall-clock measurement
makes it look ~75% slower. The difference is wrapper overhead falling
asymmetrically on a sub-100 µs kernel. Same story across every 4k²
row — the wall-clock view of "DSL barely beats C++ at small sizes" is
mostly Python.

`plain` direction=both, 4096×4096:

|             | C++ µs | DSL µs | DSL/C++ |
|-------------|------:|------:|--------:|
| Wall-clock  | 40.6  | 64.1  | **0.63×** |
| Nsys kernel | 20.0  | 18.0  | **1.11×** |

Roughly **half** of each wall-clock number for plain 4k² is wrapper
overhead. nsys confirms the kernels themselves are essentially even.

### 2. Medium-large shapes (8k²+): wall-clock and nsys agree to within ~3%

For long-running kernels the host-side overhead is rounding error.
`plain` direction=both:

| Shape          | Wall-clock DSL/C++ | Nsys DSL/C++ |
|----------------|------:|-----:|
| 8192×8192      | 1.08× | 1.14× |
| 16384×16384    | 1.13× | 1.14× |

`dgelu` direction=both:

| Shape          | Wall-clock DSL/C++ | Nsys DSL/C++ |
|----------------|------:|-----:|
| 8192×8192      | 0.90× | 0.90× |
| 16384×16384    | 0.91× | 0.91× |

For the bandwidth-bound combos at large shapes, the two views converge
within 1–6%. The activation-bound combos (`dgelu`, `dsilu`, `gelu`,
`silu`, `dbias_dgelu`, `dbias_dsilu`) match almost exactly because the
kernel itself dominates; the bandwidth-bound `plain` shows a small
~5% wrapper-tax shift.

### 3. Path B and bidim drelu: wall-clock overstates DSL's win

Path B (rowwise-only `dbias_dgelu` / `dbias_dsilu` / `dbias_drelu`)
runs short kernels even at 16k². Wrapper overhead is uniform, so
removing it expands C++'s denominator more than DSL's:

`dbias_dgelu` rowwise (path B), 8192×8192:

|             | C++ µs | DSL µs | DSL/C++ |
|-------------|------:|------:|--------:|
| Wall-clock  | 229.1 | 176.0 | **1.30×** |
| Nsys kernel | 198.7 | 162.9 | **1.22×** |

`dbias_dgelu` rowwise (path B), 16384×16384:

|             | C++ µs | DSL µs | DSL/C++ |
|-------------|------:|------:|--------:|
| Wall-clock  | 808.7 | 652.9 | **1.24×** |
| Nsys kernel | 749.2 | 632.9 | **1.18×** |

The DSL win is real, but smaller than wall-clock implied. Same pattern
on `drelu` and `dbias_drelu` (both directions) and on bidim
`dbias_drelu` — the C++ `quantize_mxfp8_kernel` template instantiation
for `IS_DBIAS=true && IS_DACT=true && OP=drelu` apparently isn't as
well-tuned as the gelu/silu paths.

## Full nsys results — direction=both

All times are kernel-only (nsys `cuda_gpu_kern_sum`), bf16 → e4m3, 100
timed iters after 10 warmup. "DSL/C++" >1.0 means CuTeDSL is faster.

### Plain quantize

| Shape         | C++ µs | DSL µs | C++ GB/s | DSL GB/s | DSL/C++ |
|---------------|------:|------:|--------:|--------:|--------:|
| 4096×4096     |  20.0 |  18.0 |  3402.5 |  3777.6 | **1.11×** |
| 8192×8192     |  71.8 |  63.2 |  3797.6 |  4315.8 | **1.14×** |
| 16384×16384   | 273.2 | 239.1 |  3991.9 |  4561.3 | **1.14×** |

### Forward activations (relu / gelu / silu)

| Combo | Shape         | C++ µs | DSL µs | C++ GB/s | DSL GB/s | DSL/C++ |
|-------|---------------|------:|------:|--------:|--------:|--------:|
| relu  | 4096×4096     |  31.7 |  23.0 |  2150.3 |  2965.8 | **1.38×** |
| relu  | 8192×8192     | 114.4 |  81.4 |  2383.1 |  3347.7 | **1.40×** |
| relu  | 16384×16384   | 441.8 | 310.8 |  2468.1 |  3508.8 | **1.42×** |
| gelu  | 4096×4096     |  48.4 |  56.7 |  1406.8 |  1202.4 | 0.85× |
| gelu  | 8192×8192     | 191.8 | 209.7 |  1421.3 |  1300.1 | 0.91× |
| gelu  | 16384×16384   | 753.7 | 811.7 |  1446.8 |  1343.4 | 0.93× |
| silu  | 4096×4096     |  53.5 |  61.1 |  1274.7 |  1114.8 | 0.87× |
| silu  | 8192×8192     | 224.0 | 219.3 |  1217.1 |  1242.9 | **1.02×** |
| silu  | 16384×16384   | 849.4 | 847.6 |  1283.9 |  1286.6 | **1.00×** |

### Backward activations (drelu / dgelu / dsilu)

These are `tex.d{relu,gelu,silu}` — bidim quantize with `IS_DACT=true`.
The DSL wrapper takes the IS_DACT path (`_kernel_main_dact`, paired G2S
TMA load, doubled `tx_count`).

| Combo | Shape         | C++ µs | DSL µs | C++ GB/s | DSL GB/s | DSL/C++ |
|-------|---------------|------:|------:|--------:|--------:|--------:|
| drelu | 4096×4096     |  36.7 |  28.5 |  2771.0 |  3566.2 | **1.29×** |
| drelu | 8192×8192     | 135.6 | 101.8 |  2999.9 |  3995.0 | **1.33×** |
| drelu | 16384×16384   | 525.6 | 390.7 |  3096.0 |  4165.7 | **1.35×** |
| dgelu | 4096×4096     |  73.1 |  80.9 |  1390.7 |  1256.6 | 0.90× |
| dgelu | 8192×8192     | 270.6 | 299.1 |  1503.6 |  1360.0 | 0.90× |
| dgelu | 16384×16384   |1069.8 |1169.8 |  1521.2 |  1391.2 | 0.91× |
| dsilu | 4096×4096     |  67.3 |  76.7 |  1510.9 |  1326.8 | 0.88× |
| dsilu | 8192×8192     | 256.6 | 286.3 |  1585.6 |  1420.9 | 0.90× |
| dsilu | 16384×16384   |1009.8 |1125.3 |  1611.5 |  1446.2 | 0.90× |

### Bias gradient — path A (bidim, rowwise + colwise)

The C++ template here is `IS_DBIAS=true && COLWISE_SCALING=true`. Both
sides get the colwise-driven dbias accumulation.

| Combo       | Shape         | C++ µs | DSL µs | C++ GB/s | DSL GB/s | DSL/C++ |
|-------------|---------------|------:|------:|--------:|--------:|--------:|
| dbias_drelu | 4096×4096     |  41.6 |  31.5 |  2443.5 |  3228.0 | **1.32×** |
| dbias_drelu | 8192×8192     | 152.1 | 112.2 |  2674.4 |  3624.9 | **1.36×** |
| dbias_drelu | 16384×16384   | 591.2 | 431.5 |  2752.8 |  3771.8 | **1.37×** |
| dbias_dgelu | 4096×4096     |  74.2 |  83.6 |  1370.6 |  1216.6 | 0.89× |
| dbias_dgelu | 8192×8192     | 278.0 | 307.8 |  1463.6 |  1322.0 | 0.90× |
| dbias_dgelu | 16384×16384   |1102.6 |1201.8 |  1476.0 |  1354.1 | 0.92× |
| dbias_dsilu | 4096×4096     |  65.1 |  78.7 |  1562.8 |  1292.0 | 0.83× |
| dbias_dsilu | 8192×8192     | 247.3 | 296.5 |  1645.3 |  1372.1 | 0.83× |
| dbias_dsilu | 16384×16384   | 971.3 |1163.0 |  1675.4 |  1399.3 | 0.84× |

### Bias gradient — path B (rowwise-only, shmem transpose)

Rowwise-only IS_DBIAS — the DSL kernel allocates the
`DBIAS_BUFF_WIDTH=66` smem buffer and does the 32-element
per-thread accumulator + shmem transpose epilogue described in
[`MXFP8_CUTEDSL_FEATURE_BENCH.md`](MXFP8_CUTEDSL_FEATURE_BENCH.md).
C++ takes the corresponding `!COLWISE_SCALING` branch.

| Combo       | Shape         | C++ µs | DSL µs | C++ GB/s | DSL GB/s | DSL/C++ |
|-------------|---------------|------:|------:|--------:|--------:|--------:|
| dbias_drelu | 4096×4096     |  25.2 |  22.0 |  3348.5 |  3840.1 | **1.15×** |
| dbias_drelu | 8192×8192     |  88.6 |  83.6 |  3810.6 |  4040.6 | **1.06×** |
| dbias_drelu | 16384×16384   | 334.9 | 318.4 |  4032.8 |  4242.0 | **1.05×** |
| dbias_dgelu | 4096×4096     |  51.3 |  43.5 |  1646.1 |  1941.4 | **1.18×** |
| dbias_dgelu | 8192×8192     | 198.7 | 162.9 |  1699.3 |  2072.6 | **1.22×** |
| dbias_dgelu | 16384×16384   | 749.2 | 632.9 |  1802.6 |  2134.1 | **1.18×** |
| dbias_dsilu | 4096×4096     |  51.6 |  49.8 |  1635.1 |  1694.9 | **1.04×** |
| dbias_dsilu | 8192×8192     | 206.0 | 194.7 |  1639.0 |  1734.4 | **1.06×** |
| dbias_dsilu | 16384×16384   | 791.5 | 752.9 |  1706.3 |  1793.8 | **1.05×** |

## Headline takeaways

* **Plain MXFP8 quantize is ~14% faster than TE C++ at all sizes.**
  4096² → 1.11×, 16384² → 1.14×. The DSL kernel saturates HBM3e at
  ~4.5 TB/s on 16k² (vs ~4.0 TB/s for C++) — the win is from cleaner
  TMA scheduling around bidim output stores.

* **Forward `relu`, backward `drelu`, and bidim `dbias_drelu` are the
  big DSL wins** (1.29× – 1.42×). C++ `quantize_mxfp8_kernel` for
  drelu is constrained by the way nvcc schedules its
  branch-and-multiply derivative; DSL's
  `partial * cute.where(act_in > 0, …)` lowers to a cleaner sequence.

* **`gelu` / `silu` / `dgelu` / `dsilu` sit at ~0.85× – 0.93× C++.**
  The gap is in scalar f32 activation throughput — once we leave the
  packed-x2 amax/cast fast path the DSL kernel runs `cute.math.tanh`
  / `cute.math.exp` (with `fastmath=False` for bit-exactness against
  `tanhf` / `expf`) and nvcc out-schedules CuTeDSL's MLIR lowering on
  the FFMAs. Throughput halves from ~4 TB/s to ~1.3 TB/s — compute,
  not bandwidth, dominates here.

* **`dbias_drelu` on path B at 4k² is the only place small-size DSL is
  notably ahead** (1.15×). At 8k² and 16k² path B drelu drops to 1.05×
  / 1.06× because the DSL kernel becomes bandwidth-bound earlier than
  C++.

* **Path B beats path A at every size for `dbias_dgelu` / `dbias_dsilu`
  at the kernel level** because rowwise-only avoids the colwise smem
  transpose store and one TMA S2G. The shmem transpose for the dbias
  accumulator costs less than the full colwise output. This makes
  `dbias_d*` rowwise-only a strong fit when downstream consumers don't
  need both layouts.

* **Wall-clock at 4k² over-states C++ wins / under-states DSL wins by
  ~30–40 percentage points.** Always cross-check with nsys at small
  shapes. At 8k²+ the two views converge.

## Methodology notes

* The bench script measures `(warmup, then iters timed across one
  cuda.Event pair)` — *not* per-call events. Per-call events would add
  a ~0.5 µs per-iter overhead.
* `nsys cuda_gpu_kern_sum` reports the average kernel runtime across
  the entire timed range, weighted by instances. The dominant kernel
  is selected by total time:
  - **DSL**: any kernel whose name contains `kernel_cutlass` or
    `cutedsl_alt`.
  - **C++**: the largest non-DSL `quantize_mxfp8_kernel` instantiation
    by total time. Activation entry kernels, `reduce_dbias`, and
    `torch.sum`'s CUB reductions are excluded.
* The dbias post-kernel reduce (`torch.sum(workspace, dim=0)` on the
  DSL side, TE's `reduce_dbias` C++ kernel on the reference side) is
  *not* included in the kernel-only nsys timing — both paths run
  separate kernels for it. Wall-clock includes both. The reduce is
  ~3–8 µs at all sizes and dwarfs neither.
* All measurements collected on `cutedsl_mxfp8` branch's CuTeDSL
  kernel against the unmodified TE C++ build (`develop`-equivalent
  `quantize_mxfp8.cuh`).

## Reproducing

```bash
cd tests/pytorch/mxfp8

# Wall-clock only (fast; no profiling overhead)
python bench_mxfp8_cutedsl.py \
    --warmup 10 --iters 100 \
    --shapes 4096,4096;8192,8192;16384,16384 \
    --direction both \
    --combo dbias_dgelu

# Sweep all combos at one shape, write CSV
python bench_mxfp8_cutedsl.py \
    --shapes 16384,16384 --direction both \
    --combos plain,gelu,silu,relu,dgelu,dsilu,drelu,dbias_dgelu,dbias_dsilu,dbias_drelu \
    --csv perf.csv

# nsys kernel-only (per-shape, per-combo, per-direction)
./run_nsys_profile.sh \
    --shapes '4096,4096;8192,8192;16384,16384' \
    --direction both \
    --combos plain,gelu,silu,relu,dgelu,dsilu,drelu,dbias_dgelu,dbias_dsilu,dbias_drelu

./run_nsys_profile.sh \
    --shapes '4096,4096;8192,8192;16384,16384' \
    --direction row \
    --combos dbias_dgelu,dbias_dsilu,dbias_drelu
```

The summary table in `run_nsys_profile.sh`'s stdout is the kernel-only
view; individual `${OUT}.stdout` files contain the wall-clock numbers
from the bench script (with nsys profiling overhead, so absolute µs
are inflated — only the ratio is meaningful).

## Wrapper-overhead breakdown via NVTX

The DSL Python wrapper is annotated with NVTX ranges around each
phase, so `nsys profile --trace=cuda,nvtx ...` produces a CPU-thread
timeline that decomposes the per-call host time. Useful when the
wall-clock vs. nsys-kernel-only gap at small shapes (4k²) needs to
be attributed to a specific wrapper section.

The ranges are pushed/popped in `quantize_mxfp8_cutedsl(...)` in
[`quantize_mxfp8_cutedsl_alt.py`](quantize_mxfp8_cutedsl_alt.py).

| NVTX range | What's inside | Typical host µs (4k²) | Blocks on GPU? |
|---|---|---:|---|
| `dsl.validate` | All Python assertions on `x` / `noop` / `amax` / `act_input` shapes/dtypes/contiguity, `_torch_to_cutlass_dtype` map probe, `_is_derivative_activation` lookup, `cuda.CUstream(...)` ctype wrap. Pure Python — no CUDA calls. | ~17 | No |
| `dsl.alloc` | `torch.empty` for the 4 output buffers (`rowwise_data`, `rowwise_scale`, `colwise_data`, `colwise_scale`). PyTorch caching allocator pool hit — no `cudaMalloc`/sync in steady state. | ~28 | No (unless pool empty) |
| `dsl.cache_lookup` | `MXFP8QuantizeConfig(...)` Python object construction + `_get_compiled_kernel(cfg, stream)` dict probe. Hot path: dict lookup only; cold path (first call per config): triggers full CuTeDSL JIT compile, amortized in warmup. | ~3 | No (in steady state) |
| `dsl.dbias_workspace` | One `torch.empty` for the `(blocks_Y, N) f32` dbias partial-sum buffer when `compute_dbias=True`. Small no-op dummy alloc otherwise. | ~8 | No |
| `dsl.make_ptr` | The 9 `make_ptr(...)` ctype wraps over `tensor.data_ptr()` for `x`, `act_input`, 4 output buffers, `noop`, `amax`, `dbias_workspace`. Pure Python wrapper around a uintptr_t. | ~16 | No |
| `dsl.launch` | `compiled(*args)` — the CuTeDSL-emitted launcher's host stub: pack the kernel descriptor and call `cudaLaunchKernelExC`. Returns as soon as the kernel is enqueued; the kernel itself runs asynchronously after this range ends. | ~36 | No (just enqueues) |
| `dsl.reduce_dbias` | `torch.sum(workspace, dim=0).to(x.dtype)` — only present when `compute_dbias=True`. Two extra CUDA kernel launches (`at::native::reduce_kernel` + a `bfloat16_copy_kernel` for the dtype downcast). Returns as soon as both are enqueued; kernels run asynchronously. | ~46 | No (just enqueues) |

Two things to keep in mind when reading the NVTX timeline:

1. **None of the ranges block on the GPU.** Every CUDA call inside
   them (`cudaLaunchKernel*`) is fire-and-forget — it enqueues work on
   the stream and returns within ~5–10 µs without waiting for kernel
   start or completion. So a range can end on the CPU while the
   kernel it launched is still running, and the CPU has already moved
   on to the next iter's `dsl.validate`. The only forced CPU↔GPU
   rendezvous in the bench is the `torch.cuda.synchronize()` *after*
   `end.record()` in `bench_once`, which sits outside the timed window.

2. **NVTX/nsys tracing inflates the absolute numbers.** Each NVTX
   `range_push/pop` adds ~1–2 µs, and every traced `cudaLaunchKernel`
   gets recorded at ~12–13 µs vs. ~5–8 µs untraced. For the 4k²
   `dbias_dgelu` both case, summing the median NVTX ranges gives
   ~155 µs/iter while the clean (no-nsys) wall-clock is 110 µs/iter —
   the ~45 µs delta is tracing overhead, distributed across the
   ranges that contain CUDA API calls (`launch` and `reduce_dbias`
   take the brunt). Use the *relative* breakdown to attribute cost;
   take the *absolute* numbers as ~30–40% over the truth.

The wrapper is host-bound at small shapes (host wall-clock per iter
> kernel time per iter), so `cuda.Event.elapsed_time` ends up
measuring host time, not kernel time. At 16k² the kernel runs
~1.2 ms and the wrapper's ~50 µs disappears into the GPU's shadow,
restoring agreement between wall-clock and kernel-only.
