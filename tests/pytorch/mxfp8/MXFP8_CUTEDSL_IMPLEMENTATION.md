# MXFP8 Quantization: CuTeDSL Implementation

## Overview

This document describes the CuTeDSL reimplementation of the MXFP8 quantization
kernel originally written in CUDA C++ (`quantize_mxfp8.cuh`). It covers what
is implemented, how it works, and what remains to close the gap with the C++
reference.

For performance numbers, see `MXFP8_CUTEDSL_PERFORMANCE.md`.

### File Locations

| File | Role |
|------|------|
| `transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh` | Original CUDA C++ kernel |
| `transformer_engine/common/cast/mxfp8/swizzle.cuh` | GEMM-swizzled scale index helper |
| `transformer_engine/common/util/ptx.cuh` | `float_to_e8m0`, `exp2f_rcp` PTX helpers |
| `tests/pytorch/mxfp8/quantize_mxfp8_cutedsl.py` | **CuTeDSL kernel + glue code** |
| `tests/pytorch/mxfp8/test_mxfp8_quantize_cutedsl.py` | Test file (27 tests, all passing) |
| `tests/pytorch/mxfp8/bench_mxfp8_cutedsl.py` | Benchmark vs. C++ reference |
| `tests/pytorch/mxfp8/run_nsys_profile.sh` | nsys profiling wrapper |
| `tests/pytorch/mxfp8/MXFP8_CUTEDSL_PERFORMANCE.md` | Performance results |

### Evolution

The implementation was built up in seven commits, each matching a C++-kernel
optimization and verified against the reference with `atol=0, rtol=0`:

```
03d46ee  Step 7: Output smem staging with coalesced write-back
0d3d17a  Step 6: cp.async global→smem loads with load/compute overlap
ac12437  Step 5: GEMM-swizzled scale layout
1d00f63  Step 4: Double-buffered shared memory pipeline
8b07e5d  Step 3: Bank-conflict-free wave access + 2-wide FP8 conversion
223a9b3  Step 2: Bidirectional single-pass kernel
0dd5204  Step 1: Shared memory tiling with C++ thread layout
```

---

## 1. What Is Implemented

The CuTeDSL implementation covers the **cast-only path** of `quantize_mxfp8.cuh`
(no activation, no dbias, no dact). Given a 2D `(M, N)` tensor in BF16, FP16,
or FP32, it produces:

- **Rowwise**: FP8E4M3 data `(M, N)` + E8M0 scales `(M, N/32)` — 32-element
  blocks along columns.
- **Colwise**: FP8E4M3 data `(M, N)` + E8M0 scales `(M/32, N)` — 32-element
  blocks along rows.
- **Bidirectional**: both of the above in a single kernel launch, input loaded
  from global memory once.
- **GEMM-swizzled scales** (optional): scales in the 128×4 tile layout expected
  by cuBLAS MXFP8 GEMMs.

### Correctness

Produces **bit-identical** output to the reference C++ kernel for all tested
shapes (128×128 up to 16384×8192), verified with `atol=0, rtol=0`:

```bash
cd tests/pytorch/mxfp8
pytest test_mxfp8_quantize_cutedsl.py           # 27 tests
```

### Constraints

- M and N must be multiples of `CHUNK_DIM = 64`.
- Input dtype: BF16, FP16, or FP32. Output dtype: FP8E4M3.
- Requires SM90+ (cp.async).

---

## 2. Kernel Structure

### 2.1 Tile Dimensions (matches C++ `quantize_mxfp8.cuh`)

| Constant           | Value | Role |
|--------------------|------:|------|
| `SCALE_DIM`        |    32 | MXFP8 block size |
| `CHUNK_DIM_Y`      |    64 | Rows per CTA |
| `CHUNK_DIM_X`      |    64 | Cols per CTA |
| `THREADS_PER_CHUNK`|    64 | Threads per CTA |
| `BUFF_DIM_Y`       |    32 | Rows per smem buffer |
| `BUFF_DIM_X`       |    64 | Cols per smem buffer |
| `STAGES`           |     2 | `CHUNK_DIM_Y / BUFF_DIM_Y` |
| `BUFFS_NUM`        |     2 | Double buffering |
| `THREADS_X`        |     2 | Rowwise scale-blocks per row (`CHUNK_DIM_X / SCALE_DIM`) |
| `THREADS_Y`        |    32 | Rowwise rows per stage |
| `PACK_SIZE`        |     4 | Elements per vector load |
| `WAVES`            |     8 | Rowwise waves per scale-block (`SCALE_DIM / PACK_SIZE`) |
| `THREADS_PER_BANK` |     4 | Threads sharing an smem bank group |

### 2.2 Single Unified Kernel

One `MXFP8QuantizeSmemKernel` class handles all three modes (rowwise-only,
colwise-only, bidirectional) via compile-time branching on `cutlass.const_expr(cfg.rowwise)`
and `cutlass.const_expr(cfg.colwise)`. Unused passes are eliminated at JIT time.

```
@cute.jit                  host-side: make_tiled_tma_atom-style setup
def __call__(...):         build gmem tensors, launch kernel
    self.kernel(...).launch(grid=..., block=..., smem=...)

@cute.kernel               device-side: GPU code
def kernel(self, ...):
    # 1. Allocate smem: 2 input buffers + 2 output buffers + scales via global
    # 2. cp.async both stages into buf 0 and buf 1 (upfront, fully overlapped)
    # 3. Stage 0: wait(1) → compute (colwise then rowwise) → TMA-like fence → store smem→gmem
    # 4. Stage 1: wait(0) → compute → fence → store
```

### 2.3 Data Flow Per Stage

```
                global memory
                     |
                cp.async (128-bit, vectorised, async)
                     v
              input smem buffer [0 or 1]  (32 x 64 x dtype)
                     |
        colwise read (stride BUFF_DIM_X, 1 col per thread)
        rowwise read (PACK_SIZE=4 waves, bank-group swizzled)
                     |
                [amax, E8M0 scale, inv_scale, mul, cast to FP8]
                     |
                  output smem buffer (32 x 64 x uint8)
                     |
             fence + barrier
                     |
              cooperative smem → gmem store (coalesced)
                     |
                global memory
```

Scales are written directly to global memory (they are small and scattered, so
smem staging would not help).

---

## 3. Optimizations Applied (Steps 1–7)

### Step 1 — Shared memory tiling with C++ thread layout

- Allocate `(32, 64)` smem tile per block, `(64, 64)` chunk, processed in 2 stages.
- Cooperative `(64 threads × 32 rows)` coalesced load from global.
- Rowwise thread mapping `tid_Y = tidx // 2`, `tid_X = tidx % 2` (matches C++).
- Colwise thread mapping `tidx → column` (matches C++).

**Effect**: halves global reads (input loaded once to smem, read twice for amax + scale).

### Step 2 — Bidirectional single-pass kernel

- Colwise and rowwise passes share the same smem tile.
- Compile-time `cutlass.const_expr` selects which passes run.

**Effect**: halves global reads again for the `rowwise + colwise` case —
one input load serves both quantization directions.

### Step 3 — Bank-conflict-free wave access + 2-wide FP8 conversion

Rowwise smem reads use the C++ kernel's bank-group swizzle to avoid 4-way bank
conflicts within a warp:

```
bank_group = (tidx % 32) // THREADS_PER_BANK   # THREADS_PER_BANK = 4
for w in range(WAVES):                         # WAVES = 8
    swizzled_grp = ((w + bank_group) * PACK_SIZE) % SCALE_DIM
    # read 4 contiguous elements at col_start + swizzled_grp
```

Also added `cvt_f32x2_to_fp8e4m3x2()` that uses the 2-wide PTX instruction
`cvt.rn.satfinite.e4m3x2.f32` to convert two floats per instruction (mirrors
`ptx::mul_cvt_2x` in the C++ kernel).

**Effect**: eliminates smem bank conflicts; halves FP8 conversion instruction count.

### Step 4 — Double-buffered smem pipeline

- Two smem input buffers (`sX0`, `sX1`) and two output buffers (`sO_row`, `sO_col`).
- Stage ping-pong pattern: stage 0 computes from `sX0` while stage 1 loads into `sX1`.

**Effect**: enables overlap between stages (realized once async loads are in place at Step 6).

### Step 5 — GEMM-swizzled scale layout

Translated `gemm_swizzled_scale_idx` from `swizzle.cuh` to a `@dsl_user_op`:

```python
@dsl_user_op
def gemm_swizzled_scale_idx(i, j, num_tiles_X):
    # 128×4 tile layout per cuBLAS MXFP8 spec
    # idx = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4 + col_in_tile
```

Exposed via `quantize_mxfp8_cutedsl(..., with_gemm_swizzled_scales=True)`.

**Effect**: scales are GEMM-ready without a separate permutation pass.

### Step 6 — cp.async global→smem loads

Replaced synchronous global loads with CuTe's async TiledCopy:

```python
thr_layout = cute.make_ordered_layout((8, 8), order=(1, 0))  # 64 threads
val_layout = cute.make_ordered_layout((4, 8), order=(1, 0))  # 8-elem vec
copy_atom_async = cute.make_copy_atom(
    cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type,
    num_bits_per_copy=128,
)
tiled_copy = cute.make_tiled_copy_tv(copy_atom_async, thr_layout, val_layout)

# Issue both stages upfront, wait for stage 0 while stage 1 loads in background
cute.copy(copy_atom_async, tXgX_s0, tXsX_s0); cute.arch.cp_async_commit_group()
cute.copy(copy_atom_async, tXgX_s1, tXsX_s1); cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(1)   # wait for stage 0, stage 1 keeps loading
cute.arch.barrier()
# ... compute stage 0 ...
cute.arch.cp_async_wait_group(0)
cute.arch.barrier()
# ... compute stage 1 ...
```

**Effect**: true async load/compute overlap. This is the most impactful perf win.

### Step 7 — Output smem staging with coalesced write-back

FP8 output written to shared memory first, then cooperatively flushed to
global memory via coalesced stores (64 threads × 1 byte/thread/row).

```
compute → sO_row / sO_col  (swizzled writes for rowwise)
fence + barrier
cooperative store: mO_row[base_row + r, block_off_X + tidx] = sO_row[r, tidx]
```

**Effect**: eliminates scattered global stores (especially the column-strided
colwise writes); matches the C++ kernel's output path structure.

---

## 4. Low-Level DSL Operations

Six `@dsl_user_op` helpers implement math not available as high-level CuTeDSL
ops:

| Function | Purpose | Implementation |
|----------|---------|----------------|
| `_bitcast_f32_to_i32` / `_bitcast_i32_to_f32` | FP32↔INT32 bitcast | `arith.bitcast` |
| `fabs_f32` | Absolute value | bitcast, clear sign bit |
| `float_to_e8m0` | Float → E8M0 biased exponent | Branchless: `(bits + 0x7FFFFF) >> 23`, clamped to 254 |
| `exp2f_rcp` | `2^(127 - biased_exp)` | `(254 - exp) << 23` as float bits + edge-case `arith.select` |
| `cvt_f32_to_fp8e4m3` | Scalar FP32 → FP8 | Inline PTX `cvt.rn.satfinite.e4m3x2.f32` with `0.0` companion |
| `cvt_f32x2_to_fp8e4m3x2` | 2-wide FP32 → FP8 | Same PTX, both operands used |
| `gemm_swizzled_scale_idx` | Compact → swizzled scale index | `swizzle.cuh` translation |

### Design notes

**`float_to_e8m0`**: The C++ version uses three branches (mantissa check,
exponent cap, subnormal edge). The CuTeDSL version uses a branchless integer
trick: adding `0x7FFFFF` (all-ones mantissa) to the IEEE 754 bit pattern
causes a carry into the exponent field exactly when any mantissa bit is set,
naturally rounding up non-power-of-2 values.

**`cvt_f32_to_fp8e4m3`**: CuTeDSL's `TensorSSA.to(Float8E4M3FN)` works on
vectors but not on scalars (the underlying MLIR op `nvgpu.cvt_fptrunc`
requires a vector operand). The scalar path uses inline PTX; the 2-wide
variant packs two real operands for full utilization of the PTX instruction.

---

## 5. What's Still Missing vs. the C++ Version

### 5.1 TMA bulk transfers (the remaining ~2× performance gap)

The C++ kernel uses `cp.async.bulk.tensor.2d` for both input loads (`gmem→smem`)
and output stores (`smem→gmem`). The CuTeDSL implementation uses Ampere-style
`cp.async` for loads and cooperative scalar stores for outputs.

**Why this matters**: TMA bulk transfers one whole 32×64 tile per PTX
instruction, bypass L1, don't occupy thread registers during transfer, and
saturate HBM bandwidth. cp.async is limited to 128-bit transfers per thread
per instruction.

**Impact**: see `MXFP8_CUTEDSL_PERFORMANCE.md` — CuTeDSL peaks at ~2.96 TB/s
rowwise while C++ reaches ~5.81 TB/s on GB200.

**Integration status**: An attempt got as far as clean compilation (0.2s with
`CUTLASS_DSL_SM_ARCH=sm_100a`) but crashed at runtime with an opaque TMA
instruction error. Key gotchas discovered:

- Pass the **TMA coord tensor** (returned from `make_tiled_tma_atom`) — not
  the original `mX` — through `flat_divide` for `tma_partition`.
- `cute.copy` with a TMA atom requires `with cute.arch.elect_one():` — a
  plain `if tidx == 0:` triggers an infinite MLIR compilation loop.
- Without `CUTLASS_DSL_SM_ARCH=sm_100a`, PTX targets SM90 and TMA instructions
  fail at runtime with "unspecified launch failure".
- The mbarrier expected-tx bytes, `mbarrier_init_fence`, and `fence_proxy`
  ordering all need to be right; debugging requires Nsight Compute.

### 5.2 Activation / Bias-Gradient Fusion

The C++ kernel templates over `IS_ACT`, `IS_DACT`, `IS_DBIAS` to fuse
activation functions (SiLU, GeLU, ...), their derivatives, and bias-gradient
accumulation into the quantization pass. In bidirectional mode with
`IS_CACHED_ACT_OP`, computed activations from the colwise pass are cached in
smem and reused in the rowwise pass.

**CuTeDSL status**: Cast-only. Adding fusion requires extra input tensors
(activation input, bias workspace) and compile-time `cutlass.const_expr`
branching in `_compute_stage`. Mechanically straightforward but untouched.

### 5.3 Amax Tracking

Optional `amax_ptr` in C++ collects the global max|x| via warp reduction +
`atomicMaxFloat`. CuTeDSL version doesn't emit an amax. Easy addition
(warp reduce via `cute.arch.warp_reduction` + an inline-PTX atomic-max).

---

## 6. How to Close the TMA Gap

Minimal change — replace `cpasync.CopyG2SOp` with `cpasync.CopyBulkTensorTileG2SOp`
for loads and `cpasync.CopyBulkTensorTileS2GOp` for stores, using the patterns
from the Blackwell FMHA example (`examples/python/CuTeDSL/blackwell/fmha.py`):

```python
# Host side
tma_atom_in, tma_tensor_in = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(), mX, smem_layout, (BUFF_DIM_Y, BUFF_DIM_X))
tma_atom_out, tma_tensor_out = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileS2GOp(), mO, smem_layout, (BUFF_DIM_Y, BUFF_DIM_X))

# Device side — prefetch descriptors at kernel entry
cpasync.prefetch_descriptor(tma_atom_in)
cpasync.prefetch_descriptor(tma_atom_out)

# Partition (pass the TMA coord tensor, flat_divided, grouped)
gX_fd = cute.flat_divide(tma_tensor_in, tiler)
tIsI, tIgI = cpasync.tma_partition(
    tma_atom_in, 0, cute.make_layout(1),
    cute.group_modes(sX_staged, 0, 2),   # staged smem
    cute.group_modes(gX_fd, 0, 2),
)

# Issue TMA load (MUST be in elect_one, not plain `if tidx == 0`)
with cute.arch.elect_one():
    cute.arch.mbarrier_arrive_and_expect_tx(mbar, tma_bytes)
    cute.copy(tma_atom_in,
              tIgI[None, bidy, bidx],
              tIsI[None, stage_idx],
              tma_bar_ptr=mbar)
cute.arch.mbarrier_wait(mbar, phase=0)

# ... compute ...

# Issue TMA store
cute.arch.fence_proxy("async.shared", space="cta")
cute.arch.barrier()
with cute.arch.elect_one():
    cute.copy(tma_atom_out, tOsO[None, stage_idx], tOgO[None, bidy, bidx])
    cute.arch.cp_async_bulk_commit_group()
cute.arch.cp_async_bulk_wait_group(0, read=True)
```

Launch environment:
```bash
CUTLASS_DSL_SM_ARCH=sm_100a python bench_mxfp8_cutedsl.py ...
```

Expected outcome if the runtime crash is resolved: rowwise bandwidth ~5.5
TB/s (≈95% of C++).

---

## 7. API

```python
from quantize_mxfp8_cutedsl import quantize_mxfp8_cutedsl

result = quantize_mxfp8_cutedsl(
    x,                                    # (M, N) BF16/FP16/FP32 CUDA tensor
    fp8_dtype="e4m3",                     # or "e5m2"
    rowwise=True,
    colwise=False,
    with_gemm_swizzled_scales=False,
)
# result["rowwise_data"]     (M, N)       uint8 (FP8)
# result["rowwise_scale"]    (M, N/32)    uint8 (E8M0)
# result["colwise_data"]     (M, N)       uint8 (FP8)
# result["colwise_scale"]    (M/32, N)    uint8 (E8M0)
```

Internally: one JIT compile per `(dtype, M, N, fp8_dtype, rowwise, colwise,
with_gemm_swizzled_scales)` tuple, cached in a module-level dict. Subsequent
calls only wrap pointers with `make_ptr` and invoke the compiled kernel.
