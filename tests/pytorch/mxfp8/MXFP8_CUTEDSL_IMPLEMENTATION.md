# MXFP8 Quantization: CuTeDSL Implementation

## Overview

This document describes the CuTeDSL reimplementation of the MXFP8 quantization
kernel originally written in CUDA C++ (`quantize_mxfp8.cuh`). It covers what
has been implemented, how it works, what optimizations from the C++ version are
missing, and concrete guidance on how to add them using CuTeDSL APIs.

### File Locations

| File | Role |
|------|------|
| `transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh` | Original CUDA C++ kernel |
| `transformer_engine/common/cast/mxfp8/swizzle.cuh` | GEMM-swizzled scale index helper |
| `transformer_engine/common/cast/mxfp8/specialized/quantize_mxfp8.cuh` | Blackwell-specialized variants |
| `transformer_engine/common/util/ptx.cuh` | `float_to_e8m0`, `exp2f_rcp` PTX helpers |
| `tests/pytorch/mxfp8/quantize_mxfp8_cutedsl.py` | **CuTeDSL kernel + glue code** |
| `tests/pytorch/mxfp8/test_mxfp8_quantize_cutedsl.py` | **Test file (27 tests, all passing)** |

---

## 1. What Is Implemented

The CuTeDSL implementation covers the **cast-only path** of `quantize_mxfp8.cuh`
(no activation, no dbias, no dact). Given a 2D `(M, N)` tensor in BF16, FP16,
or FP32, it produces:

- **Rowwise**: FP8E4M3 data `(M, N)` + E8M0 scales `(M, N/32)` — 32-element
  blocks along columns.
- **Colwise**: FP8E4M3 data `(M, N)` + E8M0 scales `(M/32, N)` — 32-element
  blocks along rows.

### Correctness

The implementation produces **bit-identical** output to the reference C++ kernel
for all tested shapes (128x128 up to 16384x8192), verified with `atol=0, rtol=0`
comparisons against `MXFP8Quantizer`.

---

## 2. How It Works

### 2.1 Algorithm (Per 32-Element Block)

```
amax = max(|x_i|)  for i in [0..31]
biased_exponent = float_to_e8m0(amax * (1 / max_fp8_value))
inverse_scale = 2^(127 - biased_exponent)
for each element:
    out[i] = fp8(x[i] * inverse_scale)
```

This is identical to the C++ kernel's logic at lines 218-297 (colwise) and
300-460 (rowwise) of `quantize_mxfp8.cuh`.

### 2.2 CuTeDSL Kernel Structure

Each kernel follows the standard CuTeDSL two-decorator pattern:

```
class MXFP8RowwiseQuantizeKernel:
    @cute.jit           # Host-side: creates CuTe tensors, launches kernel
    def __call__(self, x_ptr, out_ptr, scale_ptr, M, max_norm_rcp, stream):
        mX = cute.make_tensor(x_ptr, cute.make_layout((M, N), stride=(N, 1)))
        ...
        self.kernel(mX, mO, mS, max_norm_rcp).launch(grid=..., block=...)

    @cute.kernel         # Device-side: GPU code
    def kernel(self, mX, mO, mS, max_norm_rcp):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        ...
```

**Compilation** is done once per `(dtype, M, N, fp8_dtype)` combination via
`cute.compile()` and cached in a Python dict. At call time, only the `make_ptr`
wrapping and the compiled function invocation happen.

### 2.3 Low-Level DSL Operations

Five `@dsl_user_op` functions implement the math using MLIR arith operations
and inline PTX:

| Function | Implementation | C++ Equivalent |
|----------|---------------|----------------|
| `fabs_f32(val)` | `bitcast` to i32, clear sign bit, `bitcast` back | `fabsf()` |
| `float_to_e8m0(val)` | `(bitcast(val) + 0x7FFFFF) >> 23`, clamp to 254 via `arith.minsi` | `ptx::float_to_e8m0()` |
| `exp2f_rcp(biased_exp)` | `bitcast((254 - exp) << 23)` with `arith.select` for edges | `ptx::exp2f_rcp()` |
| `cvt_f32_to_fp8e4m3(val)` | Inline PTX: `cvt.rn.satfinite.e4m3x2.f32` | `static_cast<OType>()` |
| `_bitcast_f32_to_i32` / `_bitcast_i32_to_f32` | `arith.bitcast` | `__float_as_int` / `__int_as_float` |

**Key design note on `float_to_e8m0`**: The C++ version uses 3 branches
(mantissa check, exponent cap, subnormal edge). The CuTeDSL version uses a
branchless integer trick: adding `0x7FFFFF` (all-ones mantissa) to the IEEE 754
bit pattern causes a carry into the exponent field exactly when any mantissa bit
is set, naturally rounding up non-power-of-2 values.

**Key design note on `cvt_f32_to_fp8e4m3`**: CuTeDSL's `TensorSSA.to(Float8E4M3FN)`
works on vectors but not on scalars (the underlying MLIR op `nvgpu.cvt_fptrunc`
requires a vector operand). We work around this with inline PTX that packs
`(0.0, val)` into the 2-wide `cvt.rn.satfinite.e4m3x2.f32` instruction and
extracts the low byte.

### 2.4 Thread Mapping

**Rowwise kernel**: Each thread owns one `(row, scale_block)` pair. The thread
reads 32 contiguous elements from that row, finds the amax, computes the scale,
and writes 32 FP8 bytes + 1 scale byte.

```
Grid:  (ceil(M / rows_per_block), ceil(num_scale_cols / threads_per_row), 1)
Block: (128, 1, 1)

thread -> (global_row, scale_col) = (bidx * rows_per_block + tidx // tpr,
                                     bidy * tpr + tidx % tpr)
```

**Colwise kernel**: Each thread owns one `(scale_block, col)` pair. The thread
reads 32 elements along the column (stride = N), finds the amax, and writes
the same way.

```
Grid:  (ceil(N / 128), M // 32, 1)
Block: (128, 1, 1)
```

---

## 3. What Is Missing vs. the C++ Version

### 3.1 Shared Memory Tiling + TMA

**C++ behavior**: The kernel loads `CHUNK_DIM_Y x CHUNK_DIM_X` (64x64 or
128x128) tiles from global memory into shared memory using TMA
(`cp_async_bulk_tensor_2d`). Threads then read from shared memory.

**Current CuTeDSL**: Each thread reads 32 elements directly from global memory
via scalar loads. No shared memory is used.

**Impact**: Lower memory bandwidth utilization. TMA provides asynchronous bulk
transfers that overlap with computation and require no explicit address
calculation.

### 3.2 Double Buffering (Multi-Stage Pipeline)

**C++ behavior**: Uses `BUFFS_NUM=2` shared memory buffers and `STAGES =
CHUNK_DIM_Y / BUFF_DIM_Y` loop iterations. While stage `s` is being computed,
stage `s+1`'s TMA load is already in flight. Barrier-based synchronization
(`mbarrier`) coordinates the pipeline.

**Current CuTeDSL**: No pipelining. Each of the 32 loads is a blocking scalar
read.

**Impact**: Compute and memory are fully serialized; no latency hiding.

### 3.3 Bank-Conflict Avoidance (Swizzled Shared Memory Access)

**C++ behavior**: Rowwise reads from shared memory use a swizzled index:
```cpp
const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
```
where `bank_group = thread_lane / THREADS_PER_BANK`. This rotates which 4-byte
group each thread accesses, ensuring threads in the same cycle hit different
banks.

**Current CuTeDSL**: Not applicable (no shared memory).

### 3.4 Vectorized Loads and Stores

**C++ behavior**: Threads load `PACK_SIZE=4` elements at a time (`Vec<IType,4>.load_from()`),
process 2-wide values using FP16x2 / BF16x2 types (`IType2`, `ptx::abs_max_2x`,
`ptx::mul_cvt_2x`), and store 4 elements at a time.

**Current CuTeDSL**: All loads and stores are scalar (1 element per instruction).
The FP8 conversion uses the 2-wide `e4m3x2` PTX instruction but wastes the second
slot by passing 0.0.

**Impact**: ~4x fewer memory transactions than optimal.

### 3.5 Bidirectional Single-Pass Kernel

**C++ behavior**: When `ROWWISE_SCALING && COLWISE_SCALING`, both directions
are computed in a single kernel launch. The colwise pass reads from shared memory
first (column-strided access), then the rowwise pass reads the same shared memory
tile (row-strided access). The input is loaded from global memory only once.

**Current CuTeDSL**: Separate kernel launches for rowwise and colwise,
doubling global memory traffic.

**Impact**: 2x the global memory reads for the bidirectional case.

### 3.6 GEMM-Swizzled Scale Layout

**C++ behavior**: When `WITH_GEMM_SWIZZLED_SCALES=true`, scale factors are
written in the "swizzled" order expected by cuBLAS MXFP8 GEMMs
([docs](https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout)):
```cpp
scale_idx = gemm_swizzled_scale_idx(i, j, num_tiles_X);
// Tiles of 128 x 4 in the scale buffer, internal 32x16+4 layout
```

**Current CuTeDSL**: Scales are written in plain row-major order.

**Impact**: A separate permutation pass is needed before GEMM, or the GEMM must
fall back to non-swizzled scales.

### 3.7 Activation / Bias-Gradient Fusion

**C++ behavior**: The kernel templates over `IS_ACT`, `IS_DACT`, `IS_DBIAS` to
fuse activation functions (e.g. SiLU, GeLU), their derivatives, and bias
gradient accumulation into the quantization pass. In the bidirectional case with
activations (`IS_CACHED_ACT_OP`), computed activations from the colwise pass are
cached in shared memory and reused in the rowwise pass to avoid recomputation.

**Current CuTeDSL**: Cast-only; no activation, derivative, or bias fusion.

### 3.8 Amax Tracking

**C++ behavior**: An optional `amax_ptr` collects the global maximum absolute
value across the entire tensor using warp-level `reduce_max` followed by
`atomicMaxFloat`.

**Current CuTeDSL**: No amax output.

---

## 4. How to Implement the Missing Optimizations in CuTeDSL

### 4.1 Shared Memory Tiling

Use CuTeDSL's `SmemAllocator` to allocate a tile in shared memory, load from
global using a `TiledCopy`, then read from smem into registers.

```python
@cute.kernel
def kernel(self, mX, mO, mS, max_norm_rcp):
    cfg = self.cfg
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # --- Allocate shared memory ---
    smem = utils.SmemAllocator()
    sX = smem.allocate_tensor(
        mX.element_type,
        cute.make_ordered_layout((BUFF_DIM_Y, BUFF_DIM_X), order=(1, 0)),
        byte_alignment=128,
    )

    # --- Tile input and partition across threads ---
    tiler_mn = (BUFF_DIM_Y, BUFF_DIM_X)
    gX = cute.local_tile(mX, tiler_mn, (bidx, bidy))

    # Create TV layout (threads mapped to coalesced columns)
    thr_layout = cute.make_ordered_layout(
        (THREADS_Y, THREADS_X), order=(1, 0)
    )
    val_layout = cute.make_ordered_layout(
        (1, vec_size), order=(1, 0)
    )
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mX.element_type)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    thr_copy = tiled_copy.get_slice(tidx)
    tXgX = thr_copy.partition_S(gX)   # global source
    tXsX = thr_copy.partition_D(sX)   # smem destination

    # --- Load global -> smem ---
    cute.copy(copy_atom, tXgX, tXsX)
    cute.arch.barrier()

    # --- Read from smem into registers, compute ---
    # (thread indexing into sX for the 32-element blocks follows)
    ...
```

The key CuTeDSL concepts:
- `cute.local_tile(tensor, tile_shape, coord)` slices the global tensor to a
  per-CTA tile.
- `cute.make_tiled_copy_tv` maps threads to memory locations for coalesced
  access.
- `cute.copy(atom, src, dst)` emits vectorized load/store instructions.
- `cute.arch.barrier()` is `__syncthreads()`.

### 4.2 Double Buffering with Async Copy

For SM80+ (Ampere), use `cpasync` copy atoms. For SM90+ (Hopper), use TMA.

**Ampere async copy (cp.async):**

```python
copy_atom_async = cute.make_copy_atom(
    cute.nvgpu.cpasync.CopyG2SOp(),
    mX.element_type,
    num_bits_per_copy=128,
)

# Issue async copy for stage 0
cute.copy(copy_atom_async, tXgX_stage0, tXsX_buf0)
cute.arch.cp_async_commit_group()

# Start compute on stage 0 while stage 1 loads
for stage in range(STAGES):
    buf = stage % 2

    if stage + 1 < STAGES:
        # Prefetch next stage into the other buffer
        next_buf = (stage + 1) % 2
        cute.copy(copy_atom_async, tXgX_stages[stage+1], tXsX_bufs[next_buf])
        cute.arch.cp_async_commit_group()

    # Wait for current stage's data
    cute.arch.cp_async_wait_group(1 if stage + 1 < STAGES else 0)
    cute.arch.barrier()

    # Compute on buf
    ...compute(sX_bufs[buf])...

    cute.arch.barrier()
```

**Hopper TMA**: The C++ kernel uses `cp_async_bulk_tensor_2d` which maps to
CuTe's TMA descriptors. In CuTeDSL, this would use the TMA copy atoms from
`cute.nvgpu.tma`:

```python
# Host side: create TMA descriptor
tma_load = cute.make_tma_copy(
    cute.nvgpu.tma.CopyG2SOp(),
    mX,
    sX_layout,
)
# Device side: issue TMA load
cute.copy(tma_load, tXgX, tXsX, tma_bar_ptr=mbar)
```

### 4.3 Vectorized Loads and 2-Wide FP8 Conversion

Instead of scalar element access, use CuTe fragments:

```python
# After loading to smem:
tXrX = cute.make_fragment_like(tXsX)   # register fragment
cute.autovec_copy(tXsX, tXrX)          # smem -> rmem (vectorized)

x = tXrX.load().to(Float32)            # upcast to f32 TensorSSA

# Compute amax (local reduction over the fragment)
x_abs = fabs_tensorssa(x)              # element-wise abs via TensorSSA ops
local_amax = x_abs.reduce(cute.ReductionOp.MAX, init_val=Float32(0.0),
                           reduction_profile=0)

# Scale
scaled = x * inv_scale_broadcast

# Convert to FP8 and store (vectorized via TensorSSA.to())
tXrO.store(scaled.to(cutlass.Float8E4M3FN))
cute.copy(copy_atom_store, tXrO, tXgO)
```

The `TensorSSA.to(Float8E4M3FN)` path works on vectors and maps to the
`nvgpu.cvt_fptrunc` MLIR operation, which emits the packed `cvt.rn.satfinite`
PTX instruction on the full register vector. This eliminates the per-element
inline PTX workaround.

### 4.4 Bidirectional Single-Pass Kernel

Combine both passes in one kernel. After loading into shared memory:

1. **Colwise pass**: Each thread reads 32 elements along a column from smem
   (stride = `BUFF_DIM_X`), computes amax, scale, quantizes, writes colwise
   output.
2. `__syncthreads()`
3. **Rowwise pass**: Each thread reads 32 contiguous elements in a row from
   smem, computes amax, scale, quantizes, writes rowwise output.

Both passes share the same smem tile, so global memory is loaded only once.

The C++ kernel uses template booleans `ROWWISE_SCALING` / `COLWISE_SCALING` to
compile out unused passes. In CuTeDSL, use `cutlass.const_expr(cfg.rowwise)`:

```python
if cutlass.const_expr(cfg.colwise):
    # Colwise pass (reads from smem column-strided)
    ...

cute.arch.barrier()

if cutlass.const_expr(cfg.rowwise):
    # Rowwise pass (reads from smem row-contiguous)
    ...
```

### 4.5 GEMM-Swizzled Scale Layout

Translate the `gemm_swizzled_scale_idx` function to a `@dsl_user_op`:

```python
@dsl_user_op
def gemm_swizzled_scale_idx(i: Int32, j: Int32, num_tiles_X: Int32,
                             *, loc=None, ip=None) -> Int32:
    """Convert compact (i, j) to GEMM-swizzled linear index.

    Layout: 128x4 tiles, internal ordering:
        (row % 32) * 16 + (row // 32) * 4 + (col % 4)
    """
    TILE_DIM_X = Int32(4)
    TILE_DIM_Y = Int32(128)
    TILE_SIZE = Int32(512)   # 4 * 128

    tile_idx_X = j // TILE_DIM_X
    tile_idx_Y = i // TILE_DIM_Y
    idx_in_tile_X = j % TILE_DIM_X
    idx_in_tile_Y = i % TILE_DIM_Y

    idx = (tile_idx_Y * num_tiles_X + tile_idx_X) * TILE_SIZE
    idx = idx + (idx_in_tile_Y % Int32(32)) * Int32(16)
    idx = idx + (idx_in_tile_Y // Int32(32)) * Int32(4)
    idx = idx + idx_in_tile_X
    return idx
```

Then in the kernel, select the write index:

```python
if cutlass.const_expr(cfg.with_gemm_swizzled_scales):
    scale_idx = gemm_swizzled_scale_idx(global_row, scale_col, num_tiles)
    mS_flat[scale_idx] = Uint8(biased_exponent)
else:
    mS[global_row, scale_col] = Uint8(biased_exponent)
```

### 4.6 Bank-Conflict-Free Shared Memory Access

For the rowwise pass reading from shared memory, the C++ kernel rotates which
4-element group each thread accesses based on its bank group:

```cpp
swizzled_group_idx = ((wave + bank_group) * PACK_SIZE) % SCALE_DIM_X
```

In CuTeDSL, apply the same rotation when indexing into the smem tensor:

```python
lane = cute.arch.lane_idx()
bank_group = lane // THREADS_PER_BANK   # THREADS_PER_BANK = 4

for w in cutlass.range_constexpr(WAVES):  # WAVES = 32 / 4 = 8
    swizzled_group = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X
    swizzled_col = row_base_x + swizzled_group
    # Load 4 elements from smem at swizzled_col
    ...
```

Alternatively, CuTeDSL supports swizzle layouts natively via
`smem.allocate_tensor(..., swizzle=swizzle_layout)`, which can automate
bank-conflict avoidance at the layout level.

### 4.7 Activation / Bias-Gradient Fusion

This is a higher-level feature. The kernel would accept additional input tensors
(activation input, bias workspace) and template parameters. The structure
mirrors the C++ kernel's `COMPUTE_ACTIVATIONS` path:

```python
if cutlass.const_expr(cfg.is_act):
    elt = activation_op(elt)
if cutlass.const_expr(cfg.is_dact):
    act_in_elt = Float32(act_input_smem[offset])
    elt = elt * activation_deriv_op(act_in_elt)
if cutlass.const_expr(cfg.is_dbias):
    partial_dbias += elt
```

For the cached activation optimization (`IS_CACHED_ACT_OP`), the colwise pass
writes computed activations back to the input smem buffer, and the rowwise pass
reads from there instead of recomputing:

```python
# In colwise pass:
if cutlass.const_expr(cfg.cached_act):
    cached_act_smem[offset] = IType(elt)

cute.arch.barrier()

# In rowwise pass:
if cutlass.const_expr(cfg.cached_act):
    elt = Float32(cached_act_smem[offset])  # reuse, don't recompute
```

### 4.8 Amax Tracking

Use CuTeDSL's warp reduction and an atomicMax on global memory:

```python
from reduce import block_reduce

# After each thread computes its local amax:
block_amax = block_reduce(
    thread_amax, cute.arch.fmax, reduction_buffer, Float32(0.0)
)

# Thread 0 does atomic max to global
if tidx == 0:
    atomic_max_float(amax_ptr, block_amax)
```

The `atomic_max_float` can be implemented as a `@dsl_user_op` wrapping
`atom.global.max.f32` PTX, or using the `atomicMaxFloat` pattern from the
C++ kernel (CAS loop for float atomics on pre-SM90).

---

## 5. Suggested Implementation Order

1. **Shared memory tiling + async copy** (4.1 + 4.2): Biggest performance win.
   Use `cpasync.CopyG2SOp()` for Ampere, TMA for Hopper+. Follow the rmsnorm
   example at `cutlass/examples/python/CuTeDSL/blackwell/rmsnorm.py`.

2. **Vectorized loads + TensorSSA FP8 conversion** (4.3): Eliminates the
   per-element inline PTX workaround and reduces instruction count ~4x. Use
   `TensorSSA.to(Float8E4M3FN)` for the conversion.

3. **Bidirectional single-pass** (4.4): Halves global memory traffic for the
   common `rowwise=True, colwise=True` case. Requires shared memory from step 1.

4. **GEMM-swizzled scales** (4.5): Pure index math, easy to add as a
   `@dsl_user_op` once the kernel structure is in place.

5. **Bank-conflict avoidance** (4.6): Micro-optimization on top of step 1.

6. **Amax tracking** (4.8): Small addition; follow the `reduce.py` example.

7. **Activation/bias fusion** (4.7): Feature parity. Only needed when the
   quantization kernel is used in a fused backward pass.
