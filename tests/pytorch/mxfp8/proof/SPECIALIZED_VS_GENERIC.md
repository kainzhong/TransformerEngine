## MXFP8 `quantize`: generic kernel vs specialized cast-only kernel

Three builds of the same tree, profiled on one GB200 (bf16 -> e4m3). All numbers are **pure GPU kernel time** (median CUPTI duration via nsys `nvtx_kern_sum`); no host time is included.

| build | alignment fix | specialization | mapped `libtransformer_engine.so` md5 |
|---|---|---|---|
| **specialized** (stock) | yes | enabled | `229855200827` |
| **generic+fix** | yes | disabled | `f373c8223789` |
| **generic, no fix** | no (matches `main`) | disabled | `1f09bd126cc6` |

Specialization was disabled with a one-line `constexpr bool kAllowSpecialized = false` guard so that `plain` rowwise/bidimensional fall through to the generic kernel and can be compared head-to-head. That patch is experiment-only and is not part of the PR.

Both ratio columns are oriented so **higher is faster**:

- `fix vs nofix` -- what the alignment fix buys *within* the generic kernel.
- `gen+fix vs spec` -- >1.00x means the fixed generic kernel is **faster** than the specialized cast-only kernel; <1.00x means it is still behind.

### BIDIMENSIONAL

| shape | spec (us) | spec (GB/s) | gen+fix (us) | gen+fix (GB/s) | no-fix (us) | no-fix (GB/s) | fix vs nofix | gen+fix vs spec |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 8192x28672 | 149.1 | 6400 | 145.2 | 6574 | 154.3 | 6183 | 1.063x | 1.027x |
| 16384x5120 | 57.1 | 5971 | 54.6 | 6241 | 58.1 | 5864 | 1.064x | 1.045x |
| 8192x8192 | 46.9 | 5812 | 45.6 | 5972 | 48.0 | 5678 | 1.052x | 1.028x |
| 4096x14336 | 42.3 | 5639 | 40.1 | 5947 | 41.9 | 5691 | 1.045x | 1.055x |
| 2048x12288 | 20.1 | 5083 | 20.2 | 5063 | 21.0 | 4878 | 1.038x | 0.996x |
| 4096x4096 | 16.1 | 4234 | 14.4 | 4744 | 15.2 | 4475 | 1.060x | 1.120x |
| **median** | | | | | | | **1.056x** | **1.036x** |

### ROWWISE

| shape | spec (us) | spec (GB/s) | gen+fix (us) | gen+fix (GB/s) | no-fix (us) | no-fix (GB/s) | fix vs nofix | gen+fix vs spec |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 8192x28672 | 101.0 | 7048 | 107.4 | 6630 | 107.1 | 6647 | 0.997x | 0.941x |
| 16384x5120 | 38.8 | 6559 | 42.8 | 5948 | 42.7 | 5954 | 0.999x | 0.907x |
| 8192x8192 | 31.4 | 6470 | 36.0 | 5648 | 36.1 | 5628 | 1.004x | 0.873x |
| 4096x14336 | 28.0 | 6346 | 31.9 | 5582 | 31.6 | 5633 | 0.991x | 0.880x |
| 2048x12288 | 14.0 | 5461 | 16.3 | 4679 | 16.4 | 4656 | 1.005x | 0.857x |
| 4096x4096 | 10.0 | 5069 | 12.0 | 4255 | 12.0 | 4238 | 1.004x | 0.839x |
| **median** | | | | | | | **1.001x** | **0.876x** |

**Bidimensional:** the fixed generic kernel is 3.6% faster than the specialized cast-only kernel (median 1.036x), and it wins at every shape except 2048x12288, where the two tie. The alignment fix itself accounts for 1.056x of that within the generic kernel.

**Rowwise:** the generic kernel is still ~12% slower (median 0.876x), and the alignment fix does essentially nothing here (1.001x).

### Cache regime: the bidimensional result depends on L2 residency

Identical measurement, the only difference being whether L2 is flushed between iterations. `gen+fix vs spec`, higher is faster:

| direction | shape | working set | cold L2 (from HBM) | warm L2 (resident) |
|---|---|--:|--:|--:|
| BIDIMENSIONAL | 8192x28672 | 954 MB | 1.027x | 1.018x |
| BIDIMENSIONAL | 16384x5120 | 341 MB | 1.045x | 1.021x |
| BIDIMENSIONAL | 8192x8192 | 273 MB | 1.028x | 1.018x |
| BIDIMENSIONAL | 4096x14336 | 239 MB | 1.055x | 1.035x |
| BIDIMENSIONAL | 2048x12288 | 102 MB | 0.996x | 0.909x |
| BIDIMENSIONAL | 4096x4096 | 68 MB | 1.120x | 0.962x |
| ROWWISE | 8192x28672 | 712 MB | 0.941x | 0.964x |
| ROWWISE | 16384x5120 | 254 MB | 0.907x | 0.960x |
| ROWWISE | 8192x8192 | 203 MB | 0.873x | 0.928x |
| ROWWISE | 4096x14336 | 178 MB | 0.880x | 0.914x |
| ROWWISE | 2048x12288 | 76 MB | 0.857x | 0.922x |
| ROWWISE | 4096x4096 | 51 MB | 0.839x | 0.935x |

GB200's L2 is ~126 MB. The two smallest bidimensional shapes are the only ones whose working set fits, and they are exactly the ones where the ranking flips: served from L2 the kernel stops being bandwidth-bound, and the specialized kernel's larger 32x256 tile (half as many CTAs, so half the per-CTA prologue) wins on overhead. Streaming from HBM, the fixed generic kernel's tighter instruction stream wins instead. Rowwise does not flip -- the generic kernel trails in both regimes.

### Why rowwise is unaffected by the fix

Memory ops in the generated SASS for the `plain` kernel, before -> after:

| direction | before (generic `LD.E`/`ST.E`) | after (`LDS`/`STS`) | measured `fix vs nofix` |
|---|--:|--:|--:|
| BIDIMENSIONAL | 160 | 162 | 1.056x |
| ROWWISE | 32 | 34 | 1.001x |

Rowwise-only stages roughly a quarter as much data through shared memory as the other layouts, so there is far less for the fix to convert -- which is exactly what the timings show.

### Caveats

- Rowwise is a consistent loss for the generic kernel in **both** cache regimes. Bidimensional depends on L2 residency (see the section above): the generic kernel wins whenever the workload actually streams from HBM.
- GB/s is effective bytes (inputs read + FP8 data + e8m0 scales written) divided by kernel time. Smaller shapes fit in GB200's L2, so those figures exceed HBM bandwidth and are not memory-bandwidth utilisation.
- Only `plain` cast is covered: the specialized kernel does not exist for any fused (activation / dbias) variant.
- Single dtype pair (bf16 -> e4m3) and 6 shapes; wider coverage would be needed before acting on the bidimensional result.
