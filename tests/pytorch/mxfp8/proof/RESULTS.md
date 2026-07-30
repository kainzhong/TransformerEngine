### Generated SASS (all 612 `quantize_mxfp8_kernel` instantiations, sm_100a)

| | before | after |
|---|--:|--:|
| `LDS`/`STS` (direct shared) | 1,224 | **103,752** |
| `LD.E`/`ST.E` (generic address space) | **102,528** | **0** |
| `LDG`/`STG` (global) | 2,016 | 2,016 |
| total instructions | 1,438,784 | 1,400,720 |

Round-tripping the dynamic-SHMEM base pointer through `uintptr_t` and casting the *integer* back to a pointer loses the link to the `extern __shared__` object, so ptxas can no longer prove the address is in the shared window and falls back to generic address-space accesses. Computing the alignment as an offset on the original `char*` restores a 1:1 substitution -- `ST.E.U8` -> `STS.U8`, `LD.E.U16` -> `LDS.U16` -- removing 38,064 instructions of address arithmetic.

### Speedup by fusion (pure GPU kernel time)

| fusion | kernel | n | median | min | max |
|---|---|--:|--:|--:|--:|
| `bgrad_quantize` | generic | 18 | **1.090x** | 1.049x | 1.189x |
| `quantize` | generic | 6 | **1.063x** | 1.061x | 1.065x |
| `dbias_dgelu` + quantize | generic | 18 | **1.044x** | 1.003x | 1.097x |
| `gelu` + quantize | generic | 18 | **1.032x** | 0.936x | 1.082x |
| `dgelu` + quantize | generic | 18 | **1.022x** | 0.997x | 1.081x |
| `dsilu` + quantize | generic | 18 | **1.021x** | 0.946x | 1.054x |
| `quantize` (row / bidirectional) | specialized (untouched) | 12 | **0.999x** | 0.993x | 1.004x |

Across all 96 workloads that reach the changed kernel: **11385 us -> 10970 us (1.038x, 3.6% less kernel time)**. The 12 workloads that dispatch to the untouched specialized kernel are the control group and sit at 0.999x.

<details>
<summary>Full per-workload numbers (bf16 -> e4m3, GB200)</summary>

Rows tagged `(specialized)` dispatch to `quantize_mxfp8_kernel_cast_only`, a separate kernel this PR does not touch -- they are the control group. Which kernel ran is taken from the profile, not inferred.

| fusion | dir | shape | model tensor | before (us) | after (us) | speedup | before GB/s | after GB/s |
|---|---|---|---|--:|--:|--:|--:|--:|
| `dbias` | both | 4096x4096 | Llama-3-8B hidden | 19.0 | 17.4 | 1.089x | 3589 | 3908 |
| `dbias` | both | 16384x5120 | long-seq hidden | 68.1 | 62.7 | 1.085x | 5004 | 5432 |
| `dbias` | both | 4096x14336 | Llama-3-8B FFN | 50.2 | 46.4 | 1.081x | 4756 | 5139 |
| `dbias` | both | 8192x8192 | Llama-3-70B hidden | 56.5 | 52.3 | 1.080x | 4826 | 5212 |
| `dbias` | both | 8192x28672 | Llama-3-70B FFN | 177.2 | 164.9 | 1.075x | 5384 | 5787 |
| `dbias` | both | 2048x12288 | GPT-3-175B hidden | 24.2 | 22.7 | 1.066x | 4232 | 4509 |
| `dbias` | col | 4096x4096 | Llama-3-8B hidden | 13.2 | 11.6 | 1.136x | 3843 | 4366 |
| `dbias` | col | 4096x14336 | Llama-3-8B FFN | 33.6 | 30.8 | 1.093x | 5295 | 5785 |
| `dbias` | col | 2048x12288 | GPT-3-175B hidden | 17.8 | 16.4 | 1.090x | 4280 | 4665 |
| `dbias` | col | 16384x5120 | long-seq hidden | 44.9 | 41.7 | 1.076x | 5666 | 6098 |
| `dbias` | col | 8192x28672 | Llama-3-70B FFN | 112.2 | 106.9 | 1.049x | 6347 | 6659 |
| `dbias` | col | 8192x8192 | Llama-3-70B hidden | 37.6 | 35.9 | 1.049x | 5408 | 5671 |
| `dbias` | row | 4096x4096 | Llama-3-8B hidden | 17.2 | 14.5 | 1.189x | 2951 | 3508 |
| `dbias` | row | 8192x28672 | Llama-3-70B FFN | 160.2 | 136.6 | 1.173x | 4443 | 5212 |
| `dbias` | row | 4096x14336 | Llama-3-8B FFN | 45.3 | 39.3 | 1.152x | 3928 | 4524 |
| `dbias` | row | 16384x5120 | long-seq hidden | 61.2 | 53.2 | 1.150x | 4156 | 4781 |
| `dbias` | row | 8192x8192 | Llama-3-70B hidden | 50.8 | 44.3 | 1.145x | 4006 | 4588 |
| `dbias` | row | 2048x12288 | GPT-3-175B hidden | 22.4 | 19.6 | 1.140x | 3413 | 3892 |
| `dbias_dgelu` | both | 4096x4096 | Llama-3-8B hidden | 47.0 | 42.8 | 1.097x | 2165 | 2376 |
| `dbias_dgelu` | both | 8192x8192 | Llama-3-70B hidden | 165.8 | 153.2 | 1.082x | 2454 | 2656 |
| `dbias_dgelu` | both | 16384x5120 | long-seq hidden | 206.1 | 190.6 | 1.082x | 2467 | 2669 |
| `dbias_dgelu` | both | 4096x14336 | Llama-3-8B FFN | 146.6 | 135.5 | 1.081x | 2429 | 2627 |
| `dbias_dgelu` | both | 2048x12288 | GPT-3-175B hidden | 66.0 | 61.2 | 1.078x | 2313 | 2495 |
| `dbias_dgelu` | both | 8192x28672 | Llama-3-70B FFN | 569.3 | 530.3 | 1.073x | 2501 | 2685 |
| `dbias_dgelu` | col | 8192x28672 | Llama-3-70B FFN | 382.0 | 355.1 | 1.076x | 3093 | 3328 |
| `dbias_dgelu` | col | 16384x5120 | long-seq hidden | 134.8 | 130.6 | 1.032x | 3132 | 3231 |
| `dbias_dgelu` | col | 8192x8192 | Llama-3-70B hidden | 108.7 | 105.6 | 1.029x | 3105 | 3196 |
| `dbias_dgelu` | col | 4096x14336 | Llama-3-8B FFN | 95.7 | 93.1 | 1.029x | 3087 | 3175 |
| `dbias_dgelu` | col | 2048x12288 | GPT-3-175B hidden | 43.9 | 43.2 | 1.016x | 2885 | 2932 |
| `dbias_dgelu` | col | 4096x4096 | Llama-3-8B hidden | 31.2 | 31.1 | 1.003x | 2704 | 2712 |
| `dbias_dgelu` | row | 2048x12288 | GPT-3-175B hidden | 47.6 | 44.4 | 1.072x | 2658 | 2849 |
| `dbias_dgelu` | row | 4096x4096 | Llama-3-8B hidden | 33.9 | 32.1 | 1.056x | 2492 | 2631 |
| `dbias_dgelu` | row | 4096x14336 | Llama-3-8B FFN | 101.2 | 97.9 | 1.033x | 2921 | 3017 |
| `dbias_dgelu` | row | 8192x8192 | Llama-3-70B hidden | 114.8 | 111.3 | 1.032x | 2941 | 3034 |
| `dbias_dgelu` | row | 16384x5120 | long-seq hidden | 141.1 | 138.5 | 1.019x | 2991 | 3048 |
| `dbias_dgelu` | row | 8192x28672 | Llama-3-70B FFN | 383.4 | 379.9 | 1.009x | 3083 | 3111 |
| `dgelu` | both | 8192x28672 | Llama-3-70B FFN | 553.6 | 547.9 | 1.010x | 2572 | 2599 |
| `dgelu` | both | 16384x5120 | long-seq hidden | 201.2 | 199.4 | 1.009x | 2528 | 2550 |
| `dgelu` | both | 8192x8192 | Llama-3-70B hidden | 161.4 | 160.4 | 1.007x | 2520 | 2537 |
| `dgelu` | both | 2048x12288 | GPT-3-175B hidden | 64.8 | 64.4 | 1.005x | 2356 | 2367 |
| `dgelu` | both | 4096x4096 | Llama-3-8B hidden | 45.9 | 46.0 | 0.999x | 2215 | 2212 |
| `dgelu` | both | 4096x14336 | Llama-3-8B FFN | 142.9 | 143.4 | 0.997x | 2490 | 2483 |
| `dgelu` | col | 8192x28672 | Llama-3-70B FFN | 353.7 | 340.4 | 1.039x | 3341 | 3472 |
| `dgelu` | col | 16384x5120 | long-seq hidden | 128.0 | 123.9 | 1.033x | 3299 | 3406 |
| `dgelu` | col | 2048x12288 | GPT-3-175B hidden | 42.6 | 41.6 | 1.022x | 2975 | 3041 |
| `dgelu` | col | 4096x14336 | Llama-3-8B FFN | 90.8 | 88.9 | 1.021x | 3254 | 3322 |
| `dgelu` | col | 8192x8192 | Llama-3-70B hidden | 103.3 | 101.2 | 1.021x | 3269 | 3337 |
| `dgelu` | col | 4096x4096 | Llama-3-8B hidden | 30.8 | 30.2 | 1.020x | 2743 | 2799 |
| `dgelu` | row | 4096x14336 | Llama-3-8B FFN | 96.4 | 89.2 | 1.081x | 3065 | 3311 |
| `dgelu` | row | 2048x12288 | GPT-3-175B hidden | 44.2 | 42.1 | 1.050x | 2866 | 3009 |
| `dgelu` | row | 16384x5120 | long-seq hidden | 129.4 | 123.3 | 1.049x | 3261 | 3423 |
| `dgelu` | row | 8192x28672 | Llama-3-70B FFN | 345.7 | 330.5 | 1.046x | 3418 | 3576 |
| `dgelu` | row | 4096x4096 | Llama-3-8B hidden | 31.4 | 30.1 | 1.043x | 2685 | 2800 |
| `dgelu` | row | 8192x8192 | Llama-3-70B hidden | 104.8 | 100.9 | 1.039x | 3222 | 3347 |
| `dsilu` | both | 2048x12288 | GPT-3-175B hidden | 63.4 | 62.0 | 1.022x | 2406 | 2460 |
| `dsilu` | both | 4096x14336 | Llama-3-8B FFN | 135.9 | 136.2 | 0.998x | 2619 | 2613 |
| `dsilu` | both | 16384x5120 | long-seq hidden | 190.1 | 191.3 | 0.993x | 2676 | 2658 |
| `dsilu` | both | 8192x8192 | Llama-3-70B hidden | 153.6 | 154.7 | 0.993x | 2649 | 2630 |
| `dsilu` | both | 8192x28672 | Llama-3-70B FFN | 520.7 | 527.8 | 0.987x | 2735 | 2698 |
| `dsilu` | both | 4096x4096 | Llama-3-8B hidden | 44.4 | 46.9 | 0.946x | 2293 | 2170 |
| `dsilu` | col | 4096x14336 | Llama-3-8B FFN | 100.1 | 97.5 | 1.027x | 2950 | 3029 |
| `dsilu` | col | 16384x5120 | long-seq hidden | 139.9 | 136.7 | 1.024x | 3016 | 3088 |
| `dsilu` | col | 2048x12288 | GPT-3-175B hidden | 46.6 | 45.6 | 1.022x | 2719 | 2779 |
| `dsilu` | col | 8192x8192 | Llama-3-70B hidden | 112.6 | 110.3 | 1.020x | 2999 | 3060 |
| `dsilu` | col | 8192x28672 | Llama-3-70B FFN | 390.1 | 382.7 | 1.019x | 3029 | 3088 |
| `dsilu` | col | 4096x4096 | Llama-3-8B hidden | 32.4 | 31.8 | 1.019x | 2604 | 2654 |
| `dsilu` | row | 8192x28672 | Llama-3-70B FFN | 366.3 | 347.6 | 1.054x | 3226 | 3399 |
| `dsilu` | row | 16384x5120 | long-seq hidden | 132.5 | 129.3 | 1.025x | 3185 | 3265 |
| `dsilu` | row | 4096x4096 | Llama-3-8B hidden | 31.8 | 31.1 | 1.024x | 2651 | 2715 |
| `dsilu` | row | 8192x8192 | Llama-3-70B hidden | 107.4 | 104.9 | 1.024x | 3143 | 3218 |
| `dsilu` | row | 2048x12288 | GPT-3-175B hidden | 45.1 | 44.1 | 1.023x | 2809 | 2873 |
| `dsilu` | row | 4096x14336 | Llama-3-8B FFN | 95.5 | 93.6 | 1.020x | 3093 | 3155 |
| `gelu` | both | 4096x14336 | Llama-3-8B FFN | 102.1 | 98.5 | 1.036x | 2336 | 2421 |
| `gelu` | both | 8192x8192 | Llama-3-70B hidden | 116.0 | 112.0 | 1.035x | 2351 | 2434 |
| `gelu` | both | 16384x5120 | long-seq hidden | 143.3 | 138.6 | 1.034x | 2378 | 2459 |
| `gelu` | both | 8192x28672 | Llama-3-70B FFN | 391.7 | 380.2 | 1.030x | 2436 | 2510 |
| `gelu` | both | 2048x12288 | GPT-3-175B hidden | 46.4 | 46.0 | 1.008x | 2203 | 2221 |
| `gelu` | both | 4096x4096 | Llama-3-8B hidden | 31.3 | 33.4 | 0.936x | 2180 | 2040 |
| `gelu` | col | 4096x4096 | Llama-3-8B hidden | 22.9 | 22.6 | 1.012x | 2221 | 2248 |
| `gelu` | col | 8192x8192 | Llama-3-70B hidden | 74.6 | 74.1 | 1.006x | 2728 | 2745 |
| `gelu` | col | 4096x14336 | Llama-3-8B FFN | 66.0 | 65.8 | 1.004x | 2696 | 2707 |
| `gelu` | col | 8192x28672 | Llama-3-70B FFN | 245.7 | 245.6 | 1.000x | 2898 | 2898 |
| `gelu` | col | 2048x12288 | GPT-3-175B hidden | 31.4 | 31.5 | 0.997x | 2428 | 2420 |
| `gelu` | col | 16384x5120 | long-seq hidden | 91.0 | 91.8 | 0.992x | 2793 | 2771 |
| `gelu` | row | 8192x28672 | Llama-3-70B FFN | 262.0 | 242.1 | 1.082x | 2718 | 2941 |
| `gelu` | row | 4096x14336 | Llama-3-8B FFN | 69.8 | 64.7 | 1.080x | 2549 | 2752 |
| `gelu` | row | 8192x8192 | Llama-3-70B hidden | 78.8 | 73.0 | 1.079x | 2582 | 2786 |
| `gelu` | row | 4096x4096 | Llama-3-8B hidden | 23.6 | 22.1 | 1.065x | 2159 | 2300 |
| `gelu` | row | 2048x12288 | GPT-3-175B hidden | 32.8 | 30.8 | 1.063x | 2329 | 2475 |
| `gelu` | row | 16384x5120 | long-seq hidden | 96.6 | 91.8 | 1.052x | 2632 | 2769 |
| `plain` | col | 4096x14336 | Llama-3-8B FFN | 31.5 | 29.6 | 1.065x | 5644 | 6013 |
| `plain` | col | 8192x8192 | Llama-3-70B hidden | 35.4 | 33.2 | 1.065x | 5750 | 6121 |
| `plain` | col | 16384x5120 | long-seq hidden | 42.9 | 40.4 | 1.063x | 5928 | 6302 |
| `plain` | col | 4096x4096 | Llama-3-8B hidden | 11.9 | 11.2 | 1.063x | 4272 | 4541 |
| `plain` | col | 8192x28672 | Llama-3-70B FFN | 111.3 | 104.8 | 1.062x | 6396 | 6796 |
| `plain` | col | 2048x12288 | GPT-3-175B hidden | 15.9 | 15.0 | 1.061x | 4792 | 5083 |
| `plain (specialized)` | both | 16384x5120 | long-seq hidden | 57.3 | 57.0 | 1.004x | 5951 | 5974 |
| `plain (specialized)` | both | 2048x12288 | GPT-3-175B hidden | 20.2 | 20.1 | 1.003x | 5071 | 5087 |
| `plain (specialized)` | both | 8192x8192 | Llama-3-70B hidden | 46.9 | 46.9 | 0.999x | 5815 | 5812 |
| `plain (specialized)` | both | 8192x28672 | Llama-3-70B FFN | 149.3 | 149.5 | 0.999x | 6389 | 6382 |
| `plain (specialized)` | both | 4096x14336 | Llama-3-8B FFN | 42.2 | 42.3 | 0.998x | 5650 | 5641 |
| `plain (specialized)` | both | 4096x4096 | Llama-3-8B hidden | 16.0 | 16.1 | 0.997x | 4255 | 4243 |
| `plain (specialized)` | row | 8192x8192 | Llama-3-70B hidden | 31.4 | 31.4 | 1.001x | 6480 | 6487 |
| `plain (specialized)` | row | 16384x5120 | long-seq hidden | 38.7 | 38.7 | 1.000x | 6567 | 6567 |
| `plain (specialized)` | row | 8192x28672 | Llama-3-70B FFN | 101.3 | 101.3 | 1.000x | 7029 | 7028 |
| `plain (specialized)` | row | 4096x4096 | Llama-3-8B hidden | 10.0 | 10.0 | 0.998x | 5086 | 5077 |
| `plain (specialized)` | row | 4096x14336 | Llama-3-8B FFN | 28.1 | 28.2 | 0.995x | 6335 | 6303 |
| `plain (specialized)` | row | 2048x12288 | GPT-3-175B hidden | 13.9 | 14.0 | 0.993x | 5474 | 5436 |

</details>

### Method

- **Pure GPU kernel time.** Every number is the median CUPTI kernel duration from nsys (`nvtx_kern_sum`), counting only the MXFP8 quantize kernel. No host time is included -- only the kernel body changed, host dispatch overlaps with GPU execution in a real step, and it vanishes under CUDA graphs. (`bgrad_quantize` also launches `reduce_dbias_kernel`; it is identical in both arms and is excluded.)
- GB200, `NVTE_CUDA_ARCHS=100a`, bf16 -> e4m3, L2 flushed before every iteration.
- Both arms are full builds of the same tree differing **only** by this hunk, and each profile records the md5 of the `libtransformer_engine.so` actually mapped into the process (`8ffbd1324647` before, `229855200827` after) -- read from `/proc/self/maps` after load, so a stale build cannot masquerade as a fresh one.
- **Control group:** `plain|row` and `plain|both` dispatch to the specialized cast-only kernel, which this change does not touch. They stay flat, confirming the harness is not manufacturing a difference.
- **Numerics:** 72 combo/shape/direction configurations hashed under both builds are **bitwise identical** (0 mismatches). This is a pure codegen change.
