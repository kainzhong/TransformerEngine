# MXFP8 / NVFP4 / FP8 shared-memory provenance fix -- results

Aligning the dynamic-SHMEM base through an integer and casting the *integer* back to a pointer loses the link to the `extern __shared__` object. ptxas can then no longer prove the address is in the shared window, and falls back to generic address-space accesses (`LD.E`/`ST.E`) that resolve the window at runtime. Deriving the aligned address from the original pointer by pointer arithmetic restores `LDS`/`STS`; the shared `align_up()` helper in `common/utils.cuh` is the single place that idiom now lives.

| | |
|---|---|
| before | `8ffbd1324647` (main, no fixes) |
| after | `5a7a00ac2e83` (all five fixes) |
| GPU | NVIDIA GB200 |
| host | `ptyche0065.ptyche.clusters.nvidia.com` (both arms; kernel times only compare within a node) |
| timing | pure GPU kernel time, median CUPTI duration via nsys `nvtx_kern_sum` |

## Summary

| fix | file | n | cold median | cold best | warm median | warm best |
|---|---|--:|--:|--:|--:|--:|
| generic | `cast/mxfp8/quantize_mxfp8.cuh` | 96 | **1.037x** | 1.202x | **1.041x** | 1.295x |
| specialized | `cast/mxfp8/specialized/quantize_mxfp8.cuh` | 12 | **1.001x** | 1.014x | **1.004x** | 1.132x |
| mxfp8_gated | `cast/mxfp8/gated_mxfp8.cuh` | 54 | **1.016x** | 1.090x | **1.018x** | 1.089x |
| nvfp4 | `cast/nvfp4/quantize_transpose_nvfp4.cuh` | 18 | **1.026x** | 1.068x | **1.023x** | 1.060x |
| fp8_gated | `cast/fp8/gated_fp8.cuh` | 24 | **1.013x** | 1.028x | **1.021x** | 1.031x |
| **all** | | **204** | **1.023x** | 1.202x | **1.025x** | 1.295x |

Total kernel time over all 204 workloads: **cold 24371 -> 23638 us (1.031x)**, **warm 24101 -> 23319 us (1.034x)**. Worst single workload: cold **0.943x**, warm **0.954x**.

## MXFP8 generic quantize kernel

`cast/mxfp8/quantize_mxfp8.cuh`

| fusion | dir | n | cold median | cold range | warm median | warm range |
|---|---|--:|--:|--:|--:|--:|
| `dbias` | row | 6 | **1.149x** | 1.133-1.173 | **1.159x** | 1.149-1.196 |
| `dbias` | col | 6 | **1.093x** | 1.055-1.202 | **1.180x** | 1.122-1.295 |
| `dbias` | both | 6 | **1.084x** | 1.035-1.144 | **1.090x** | 1.071-1.114 |
| `dbias_dgelu` | both | 6 | **1.077x** | 1.069-1.088 | **1.087x** | 1.073-1.095 |
| `gelu` | row | 6 | **1.076x** | 1.062-1.081 | **1.077x** | 1.057-1.082 |
| `plain` | col | 6 | **1.061x** | 1.050-1.079 | **1.180x** | 1.075-1.251 |
| `dgelu` | row | 6 | **1.046x** | 1.010-1.050 | **1.044x** | 1.038-1.048 |
| `gelu` | both | 6 | **1.034x** | 0.943-1.039 | **1.036x** | 0.954-1.045 |
| `dbias_dgelu` | row | 6 | **1.031x** | 1.005-1.078 | **1.031x** | 0.998-1.087 |
| `dgelu` | col | 6 | **1.026x** | 0.999-1.044 | **1.043x** | 1.031-1.078 |
| `dsilu` | row | 6 | **1.025x** | 1.016-1.048 | **1.030x** | 1.022-1.049 |
| `dsilu` | col | 6 | **1.022x** | 1.015-1.024 | **1.021x** | 1.020-1.023 |
| `dbias_dgelu` | col | 6 | **1.021x** | 1.006-1.070 | **1.032x** | 1.015-1.071 |
| `gelu` | col | 6 | **1.005x** | 1.000-1.013 | **1.002x** | 0.998-1.007 |
| `dgelu` | both | 6 | **1.002x** | 0.988-1.013 | **1.007x** | 0.991-1.037 |
| `dsilu` | both | 6 | **0.994x** | 0.984-1.021 | **1.000x** | 0.984-1.020 |

## MXFP8 specialized cast-only kernel (bidimensional)

`cast/mxfp8/specialized/quantize_mxfp8.cuh`

| fusion | dir | n | cold median | cold range | warm median | warm range |
|---|---|--:|--:|--:|--:|--:|
| `plain` | both | 6 | **1.003x** | 0.997-1.014 | **1.006x** | 0.999-1.132 |
| `plain` | row | 6 | **1.000x** | 0.998-1.002 | **1.000x** | 0.996-1.008 |

## MXFP8 gated (swiglu / geglu) kernel

`cast/mxfp8/gated_mxfp8.cuh`

| fusion | dir | n | cold median | cold range | warm median | warm range |
|---|---|--:|--:|--:|--:|--:|
| `dswiglu` | col | 6 | **1.085x** | 1.079-1.090 | **1.088x** | 1.081-1.089 |
| `geglu` | row | 6 | **1.077x** | 1.072-1.085 | **1.086x** | 1.073-1.089 |
| `geglu` | both | 6 | **1.061x** | 1.055-1.064 | **1.061x** | 1.057-1.064 |
| `dswiglu` | both | 6 | **1.022x** | 1.011-1.037 | **1.016x** | 1.011-1.031 |
| `dswiglu` | row | 6 | **1.015x** | 1.012-1.018 | **1.016x** | 1.011-1.024 |
| `swiglu` | col | 6 | **1.015x** | 1.013-1.016 | **1.015x** | 1.013-1.025 |
| `swiglu` | both | 6 | **1.009x** | 1.000-1.060 | **1.003x** | 0.999-1.045 |
| `geglu` | col | 6 | **1.006x** | 1.002-1.009 | **1.004x** | 1.000-1.009 |
| `swiglu` | row | 6 | **1.006x** | 0.987-1.016 | **1.013x** | 1.010-1.019 |

## NVFP4 2D quantize-transpose kernel

`cast/nvfp4/quantize_transpose_nvfp4.cuh`

| fusion | dir | n | cold median | cold range | warm median | warm range |
|---|---|--:|--:|--:|--:|--:|
| `nvfp42d` | row | 6 | **1.029x** | 1.022-1.031 | **1.026x** | 1.016-1.030 |
| `nvfp42d` | both | 6 | **1.025x** | 1.004-1.039 | **1.019x** | 1.005-1.043 |
| `nvfp42d` | col | 6 | **1.022x** | 1.014-1.068 | **1.023x** | 1.014-1.060 |

## FP8 gated (swiglu / geglu) kernel

`cast/fp8/gated_fp8.cuh`

| fusion | dir | n | cold median | cold range | warm median | warm range |
|---|---|--:|--:|--:|--:|--:|
| `dswiglu` | na | 6 | **1.023x** | 0.999-1.028 | **1.029x** | 1.022-1.031 |
| `swiglu` | na | 6 | **1.022x** | 1.013-1.023 | **1.021x** | 1.019-1.024 |
| `dgeglu` | na | 6 | **1.011x** | 1.009-1.013 | **1.016x** | 0.997-1.022 |
| `geglu` | na | 6 | **1.000x** | 0.999-1.007 | **1.006x** | 1.003-1.024 |

<details>
<summary>Full per-workload numbers (us, pure GPU kernel time)</summary>

| fix | fusion | dir | shape | model tensor | cold before | cold after | cold | warm before | warm after | warm |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|
| fp8_gated | `dgeglu` | na | 8192x28672 | Llama-3-70B FFN | 458.8 | 453.1 | 1.013x | 458.4 | 450.7 | 1.017x |
| fp8_gated | `dgeglu` | na | 4096x14336 | Llama-3-8B FFN | 122.5 | 121.0 | 1.012x | 121.1 | 119.1 | 1.017x |
| fp8_gated | `dgeglu` | na | 2048x12288 | GPT-3-175B hidden | 58.0 | 57.3 | 1.012x | 56.4 | 56.6 | 0.997x |
| fp8_gated | `dgeglu` | na | 8192x8192 | Llama-3-70B hidden | 137.9 | 136.4 | 1.011x | 136.2 | 134.2 | 1.015x |
| fp8_gated | `dgeglu` | na | 4096x4096 | Llama-3-8B hidden | 40.5 | 40.1 | 1.011x | 39.9 | 39.0 | 1.022x |
| fp8_gated | `dgeglu` | na | 16384x5120 | long-seq hidden | 169.4 | 168.0 | 1.009x | 166.4 | 166.9 | 0.997x |
| fp8_gated | `dswiglu` | na | 4096x14336 | Llama-3-8B FFN | 89.2 | 86.7 | 1.028x | 86.8 | 84.9 | 1.022x |
| fp8_gated | `dswiglu` | na | 8192x28672 | Llama-3-70B FFN | 321.1 | 313.0 | 1.026x | 319.2 | 310.7 | 1.027x |
| fp8_gated | `dswiglu` | na | 8192x8192 | Llama-3-70B hidden | 100.4 | 98.0 | 1.024x | 98.0 | 95.0 | 1.031x |
| fp8_gated | `dswiglu` | na | 4096x4096 | Llama-3-8B hidden | 31.6 | 30.9 | 1.022x | 29.6 | 28.8 | 1.028x |
| fp8_gated | `dswiglu` | na | 16384x5120 | long-seq hidden | 121.1 | 119.6 | 1.013x | 120.6 | 117.2 | 1.029x |
| fp8_gated | `dswiglu` | na | 2048x12288 | GPT-3-175B hidden | 42.6 | 42.6 | 0.999x | 42.0 | 40.7 | 1.031x |
| fp8_gated | `geglu` | na | 8192x8192 | Llama-3-70B hidden | 75.8 | 75.2 | 1.007x | 73.7 | 73.4 | 1.004x |
| fp8_gated | `geglu` | na | 8192x28672 | Llama-3-70B FFN | 241.3 | 240.4 | 1.004x | 239.7 | 239.1 | 1.003x |
| fp8_gated | `geglu` | na | 16384x5120 | long-seq hidden | 92.9 | 92.8 | 1.001x | 90.8 | 90.0 | 1.009x |
| fp8_gated | `geglu` | na | 4096x14336 | Llama-3-8B FFN | 67.2 | 67.2 | 1.000x | 65.5 | 64.9 | 1.009x |
| fp8_gated | `geglu` | na | 4096x4096 | Llama-3-8B hidden | 23.5 | 23.5 | 1.000x | 20.9 | 20.4 | 1.024x |
| fp8_gated | `geglu` | na | 2048x12288 | GPT-3-175B hidden | 33.0 | 33.0 | 0.999x | 31.7 | 31.6 | 1.003x |
| fp8_gated | `swiglu` | na | 8192x28672 | Llama-3-70B FFN | 244.9 | 239.4 | 1.023x | 243.1 | 237.5 | 1.024x |
| fp8_gated | `swiglu` | na | 16384x5120 | long-seq hidden | 94.2 | 92.2 | 1.022x | 92.1 | 90.3 | 1.020x |
| fp8_gated | `swiglu` | na | 8192x8192 | Llama-3-70B hidden | 76.8 | 75.2 | 1.022x | 74.7 | 73.1 | 1.021x |
| fp8_gated | `swiglu` | na | 4096x14336 | Llama-3-8B FFN | 68.1 | 66.7 | 1.022x | 66.7 | 65.3 | 1.021x |
| fp8_gated | `swiglu` | na | 4096x4096 | Llama-3-8B hidden | 23.9 | 23.5 | 1.016x | 21.1 | 20.7 | 1.019x |
| fp8_gated | `swiglu` | na | 2048x12288 | GPT-3-175B hidden | 33.4 | 33.0 | 1.013x | 32.5 | 31.8 | 1.022x |
| generic | `dbias` | both | 16384x5120 | long-seq hidden | 73.0 | 63.8 | 1.144x | 69.4 | 63.9 | 1.087x |
| generic | `dbias` | both | 4096x4096 | Llama-3-8B hidden | 19.9 | 17.6 | 1.129x | 16.5 | 14.8 | 1.112x |
| generic | `dbias` | both | 8192x28672 | Llama-3-70B FFN | 179.7 | 165.7 | 1.084x | 178.2 | 166.4 | 1.071x |
| generic | `dbias` | both | 8192x8192 | Llama-3-70B hidden | 56.9 | 52.5 | 1.084x | 57.0 | 52.2 | 1.093x |
| generic | `dbias` | both | 4096x14336 | Llama-3-8B FFN | 50.4 | 46.5 | 1.082x | 50.0 | 46.0 | 1.087x |
| generic | `dbias` | both | 2048x12288 | GPT-3-175B hidden | 23.7 | 22.9 | 1.035x | 23.5 | 21.1 | 1.114x |
| generic | `dbias` | col | 4096x4096 | Llama-3-8B hidden | 13.8 | 11.5 | 1.202x | 11.0 | 9.0 | 1.223x |
| generic | `dbias` | col | 16384x5120 | long-seq hidden | 48.2 | 41.9 | 1.148x | 45.3 | 39.0 | 1.160x |
| generic | `dbias` | col | 8192x28672 | Llama-3-70B FFN | 117.1 | 106.9 | 1.096x | 116.2 | 103.6 | 1.122x |
| generic | `dbias` | col | 4096x14336 | Llama-3-8B FFN | 33.6 | 30.8 | 1.091x | 32.8 | 27.6 | 1.187x |
| generic | `dbias` | col | 2048x12288 | GPT-3-175B hidden | 17.6 | 16.3 | 1.078x | 15.4 | 11.9 | 1.295x |
| generic | `dbias` | col | 8192x8192 | Llama-3-70B hidden | 37.6 | 35.6 | 1.055x | 37.0 | 31.5 | 1.173x |
| generic | `dbias` | row | 8192x28672 | Llama-3-70B FFN | 160.3 | 136.7 | 1.173x | 157.7 | 134.4 | 1.173x |
| generic | `dbias` | row | 4096x4096 | Llama-3-8B hidden | 17.2 | 14.7 | 1.172x | 12.8 | 10.7 | 1.196x |
| generic | `dbias` | row | 4096x14336 | Llama-3-8B FFN | 45.4 | 39.3 | 1.153x | 42.8 | 37.0 | 1.156x |
| generic | `dbias` | row | 16384x5120 | long-seq hidden | 61.1 | 53.4 | 1.144x | 59.1 | 50.9 | 1.162x |
| generic | `dbias` | row | 8192x8192 | Llama-3-70B hidden | 50.8 | 44.4 | 1.143x | 48.3 | 42.0 | 1.149x |
| generic | `dbias` | row | 2048x12288 | GPT-3-175B hidden | 22.4 | 19.8 | 1.133x | 17.1 | 14.9 | 1.149x |
| generic | `dbias_dgelu` | both | 4096x4096 | Llama-3-8B hidden | 47.0 | 43.2 | 1.088x | 46.8 | 42.7 | 1.095x |
| generic | `dbias_dgelu` | both | 4096x14336 | Llama-3-8B FFN | 146.8 | 135.6 | 1.083x | 147.9 | 136.2 | 1.086x |
| generic | `dbias_dgelu` | both | 16384x5120 | long-seq hidden | 206.4 | 191.5 | 1.078x | 207.8 | 191.8 | 1.083x |
| generic | `dbias_dgelu` | both | 8192x8192 | Llama-3-70B hidden | 166.2 | 154.4 | 1.076x | 167.8 | 154.2 | 1.088x |
| generic | `dbias_dgelu` | both | 2048x12288 | GPT-3-175B hidden | 65.7 | 61.1 | 1.076x | 66.6 | 61.2 | 1.089x |
| generic | `dbias_dgelu` | both | 8192x28672 | Llama-3-70B FFN | 568.5 | 531.8 | 1.069x | 570.8 | 531.9 | 1.073x |
| generic | `dbias_dgelu` | col | 8192x28672 | Llama-3-70B FFN | 381.3 | 356.2 | 1.070x | 381.0 | 355.6 | 1.071x |
| generic | `dbias_dgelu` | col | 16384x5120 | long-seq hidden | 135.4 | 130.6 | 1.036x | 135.0 | 129.9 | 1.039x |
| generic | `dbias_dgelu` | col | 4096x14336 | Llama-3-8B FFN | 95.9 | 93.9 | 1.021x | 95.4 | 92.5 | 1.031x |
| generic | `dbias_dgelu` | col | 8192x8192 | Llama-3-70B hidden | 108.3 | 106.1 | 1.020x | 108.9 | 105.4 | 1.033x |
| generic | `dbias_dgelu` | col | 2048x12288 | GPT-3-175B hidden | 43.7 | 43.3 | 1.008x | 43.2 | 42.6 | 1.015x |
| generic | `dbias_dgelu` | col | 4096x4096 | Llama-3-8B hidden | 31.6 | 31.4 | 1.006x | 29.0 | 28.5 | 1.019x |
| generic | `dbias_dgelu` | row | 2048x12288 | GPT-3-175B hidden | 47.7 | 44.3 | 1.078x | 47.1 | 43.6 | 1.080x |
| generic | `dbias_dgelu` | row | 4096x4096 | Llama-3-8B hidden | 33.9 | 32.3 | 1.052x | 32.1 | 29.6 | 1.087x |
| generic | `dbias_dgelu` | row | 8192x8192 | Llama-3-70B hidden | 115.4 | 111.8 | 1.033x | 114.7 | 111.5 | 1.029x |
| generic | `dbias_dgelu` | row | 4096x14336 | Llama-3-8B FFN | 101.4 | 98.6 | 1.028x | 100.9 | 97.6 | 1.033x |
| generic | `dbias_dgelu` | row | 16384x5120 | long-seq hidden | 141.2 | 138.2 | 1.022x | 140.9 | 138.8 | 1.015x |
| generic | `dbias_dgelu` | row | 8192x28672 | Llama-3-70B FFN | 382.5 | 380.6 | 1.005x | 382.2 | 382.9 | 0.998x |
| generic | `dgelu` | both | 8192x28672 | Llama-3-70B FFN | 553.2 | 545.9 | 1.013x | 553.8 | 549.1 | 1.008x |
| generic | `dgelu` | both | 4096x14336 | Llama-3-8B FFN | 143.5 | 143.1 | 1.003x | 147.6 | 142.4 | 1.037x |
| generic | `dgelu` | both | 8192x8192 | Llama-3-70B hidden | 161.9 | 161.5 | 1.003x | 162.9 | 162.0 | 1.006x |
| generic | `dgelu` | both | 16384x5120 | long-seq hidden | 201.1 | 200.6 | 1.002x | 201.5 | 200.0 | 1.008x |
| generic | `dgelu` | both | 2048x12288 | GPT-3-175B hidden | 64.4 | 64.3 | 1.002x | 64.1 | 64.1 | 1.000x |
| generic | `dgelu` | both | 4096x4096 | Llama-3-8B hidden | 45.4 | 46.0 | 0.988x | 44.2 | 44.6 | 0.991x |
| generic | `dgelu` | col | 8192x28672 | Llama-3-70B FFN | 354.2 | 339.3 | 1.044x | 354.2 | 338.1 | 1.048x |
| generic | `dgelu` | col | 4096x14336 | Llama-3-8B FFN | 91.5 | 89.1 | 1.027x | 94.3 | 87.5 | 1.078x |
| generic | `dgelu` | col | 8192x8192 | Llama-3-70B hidden | 103.9 | 101.3 | 1.026x | 102.7 | 99.6 | 1.031x |
| generic | `dgelu` | col | 2048x12288 | GPT-3-175B hidden | 42.6 | 41.5 | 1.026x | 41.8 | 40.5 | 1.033x |
| generic | `dgelu` | col | 4096x4096 | Llama-3-8B hidden | 31.0 | 30.2 | 1.024x | 29.0 | 27.7 | 1.046x |
| generic | `dgelu` | col | 16384x5120 | long-seq hidden | 127.4 | 127.6 | 0.999x | 127.2 | 122.4 | 1.039x |
| generic | `dgelu` | row | 4096x4096 | Llama-3-8B hidden | 31.6 | 30.1 | 1.050x | 29.4 | 28.1 | 1.048x |
| generic | `dgelu` | row | 8192x8192 | Llama-3-70B hidden | 105.7 | 101.0 | 1.047x | 104.0 | 100.0 | 1.040x |
| generic | `dgelu` | row | 4096x14336 | Llama-3-8B FFN | 93.3 | 89.2 | 1.047x | 92.3 | 88.3 | 1.045x |
| generic | `dgelu` | row | 2048x12288 | GPT-3-175B hidden | 44.1 | 42.1 | 1.046x | 43.3 | 41.7 | 1.038x |
| generic | `dgelu` | row | 8192x28672 | Llama-3-70B FFN | 346.0 | 331.5 | 1.044x | 344.9 | 330.7 | 1.043x |
| generic | `dgelu` | row | 16384x5120 | long-seq hidden | 129.2 | 127.9 | 1.010x | 127.9 | 122.4 | 1.045x |
| generic | `dsilu` | both | 2048x12288 | GPT-3-175B hidden | 63.1 | 61.9 | 1.021x | 62.2 | 61.0 | 1.020x |
| generic | `dsilu` | both | 4096x4096 | Llama-3-8B hidden | 44.4 | 43.7 | 1.016x | 41.7 | 41.4 | 1.007x |
| generic | `dsilu` | both | 16384x5120 | long-seq hidden | 190.2 | 191.0 | 0.996x | 188.5 | 190.2 | 0.991x |
| generic | `dsilu` | both | 4096x14336 | Llama-3-8B FFN | 135.9 | 136.8 | 0.993x | 135.0 | 135.3 | 0.998x |
| generic | `dsilu` | both | 8192x8192 | Llama-3-70B hidden | 154.2 | 155.3 | 0.993x | 153.3 | 152.9 | 1.003x |
| generic | `dsilu` | both | 8192x28672 | Llama-3-70B FFN | 520.3 | 528.7 | 0.984x | 518.8 | 527.1 | 0.984x |
| generic | `dsilu` | col | 8192x8192 | Llama-3-70B hidden | 113.4 | 110.8 | 1.024x | 112.2 | 109.7 | 1.023x |
| generic | `dsilu` | col | 2048x12288 | GPT-3-175B hidden | 46.6 | 45.5 | 1.023x | 45.1 | 44.2 | 1.020x |
| generic | `dsilu` | col | 4096x14336 | Llama-3-8B FFN | 99.7 | 97.5 | 1.023x | 98.6 | 96.7 | 1.020x |
| generic | `dsilu` | col | 16384x5120 | long-seq hidden | 139.5 | 136.6 | 1.021x | 138.8 | 135.8 | 1.022x |
| generic | `dsilu` | col | 8192x28672 | Llama-3-70B FFN | 389.9 | 382.7 | 1.019x | 388.2 | 380.6 | 1.020x |
| generic | `dsilu` | col | 4096x4096 | Llama-3-8B hidden | 32.4 | 32.0 | 1.015x | 28.2 | 27.6 | 1.023x |
| generic | `dsilu` | row | 8192x28672 | Llama-3-70B FFN | 365.1 | 348.5 | 1.048x | 364.1 | 347.0 | 1.049x |
| generic | `dsilu` | row | 16384x5120 | long-seq hidden | 133.0 | 129.2 | 1.029x | 131.5 | 127.7 | 1.030x |
| generic | `dsilu` | row | 4096x14336 | Llama-3-8B FFN | 95.5 | 93.1 | 1.025x | 94.6 | 91.8 | 1.031x |
| generic | `dsilu` | row | 8192x8192 | Llama-3-70B hidden | 107.7 | 105.2 | 1.024x | 107.2 | 103.7 | 1.033x |
| generic | `dsilu` | row | 2048x12288 | GPT-3-175B hidden | 45.1 | 44.1 | 1.024x | 43.7 | 42.8 | 1.022x |
| generic | `dsilu` | row | 4096x4096 | Llama-3-8B hidden | 31.8 | 31.3 | 1.016x | 27.5 | 26.8 | 1.023x |
| generic | `gelu` | both | 16384x5120 | long-seq hidden | 143.5 | 138.2 | 1.039x | 143.6 | 138.4 | 1.038x |
| generic | `gelu` | both | 8192x8192 | Llama-3-70B hidden | 116.2 | 111.9 | 1.038x | 116.2 | 111.2 | 1.045x |
| generic | `gelu` | both | 8192x28672 | Llama-3-70B FFN | 393.6 | 379.6 | 1.037x | 392.8 | 379.5 | 1.035x |
| generic | `gelu` | both | 4096x14336 | Llama-3-8B FFN | 101.7 | 98.6 | 1.032x | 101.7 | 98.0 | 1.037x |
| generic | `gelu` | both | 2048x12288 | GPT-3-175B hidden | 46.6 | 45.8 | 1.016x | 46.7 | 45.2 | 1.034x |
| generic | `gelu` | both | 4096x4096 | Llama-3-8B hidden | 31.2 | 33.1 | 0.943x | 29.9 | 31.4 | 0.954x |
| generic | `gelu` | col | 4096x4096 | Llama-3-8B hidden | 22.8 | 22.5 | 1.013x | 20.9 | 21.0 | 0.998x |
| generic | `gelu` | col | 2048x12288 | GPT-3-175B hidden | 31.6 | 31.3 | 1.009x | 29.7 | 29.8 | 0.998x |
| generic | `gelu` | col | 8192x8192 | Llama-3-70B hidden | 74.5 | 74.1 | 1.005x | 73.9 | 73.7 | 1.003x |
| generic | `gelu` | col | 4096x14336 | Llama-3-8B FFN | 66.0 | 65.7 | 1.004x | 65.5 | 65.2 | 1.004x |
| generic | `gelu` | col | 16384x5120 | long-seq hidden | 91.5 | 91.2 | 1.003x | 91.2 | 90.7 | 1.007x |
| generic | `gelu` | col | 8192x28672 | Llama-3-70B FFN | 245.6 | 245.5 | 1.000x | 245.1 | 245.0 | 1.001x |
| generic | `gelu` | row | 8192x28672 | Llama-3-70B FFN | 261.7 | 242.1 | 1.081x | 261.2 | 241.3 | 1.082x |
| generic | `gelu` | row | 16384x5120 | long-seq hidden | 97.1 | 90.1 | 1.077x | 96.4 | 89.2 | 1.081x |
| generic | `gelu` | row | 8192x8192 | Llama-3-70B hidden | 78.6 | 73.0 | 1.076x | 78.0 | 72.5 | 1.075x |
| generic | `gelu` | row | 4096x14336 | Llama-3-8B FFN | 69.5 | 64.6 | 1.076x | 69.2 | 64.2 | 1.079x |
| generic | `gelu` | row | 2048x12288 | GPT-3-175B hidden | 32.9 | 30.8 | 1.069x | 31.2 | 29.2 | 1.070x |
| generic | `gelu` | row | 4096x4096 | Llama-3-8B hidden | 23.6 | 22.2 | 1.062x | 21.8 | 20.6 | 1.057x |
| generic | `plain` | col | 8192x28672 | Llama-3-70B FFN | 113.0 | 104.7 | 1.079x | 110.9 | 103.2 | 1.075x |
| generic | `plain` | col | 8192x8192 | Llama-3-70B hidden | 35.4 | 33.1 | 1.072x | 34.5 | 29.7 | 1.162x |
| generic | `plain` | col | 2048x12288 | GPT-3-175B hidden | 15.9 | 14.9 | 1.063x | 13.7 | 10.9 | 1.251x |
| generic | `plain` | col | 4096x4096 | Llama-3-8B hidden | 11.8 | 11.2 | 1.059x | 9.8 | 7.9 | 1.240x |
| generic | `plain` | col | 16384x5120 | long-seq hidden | 43.0 | 40.7 | 1.057x | 42.3 | 37.2 | 1.137x |
| generic | `plain` | col | 4096x14336 | Llama-3-8B FFN | 31.4 | 29.9 | 1.050x | 30.9 | 25.8 | 1.198x |
| mxfp8_gated | `dswiglu` | both | 4096x4096 | Llama-3-8B hidden | 72.4 | 69.8 | 1.037x | 70.7 | 68.5 | 1.031x |
| mxfp8_gated | `dswiglu` | both | 2048x12288 | GPT-3-175B hidden | 105.0 | 101.5 | 1.035x | 103.0 | 100.8 | 1.022x |
| mxfp8_gated | `dswiglu` | both | 8192x8192 | Llama-3-70B hidden | 261.4 | 255.8 | 1.022x | 259.2 | 254.6 | 1.018x |
| mxfp8_gated | `dswiglu` | both | 4096x14336 | Llama-3-8B FFN | 230.7 | 225.9 | 1.021x | 228.0 | 225.0 | 1.013x |
| mxfp8_gated | `dswiglu` | both | 16384x5120 | long-seq hidden | 322.2 | 317.9 | 1.014x | 319.6 | 315.6 | 1.012x |
| mxfp8_gated | `dswiglu` | both | 8192x28672 | Llama-3-70B FFN | 886.5 | 877.1 | 1.011x | 882.8 | 873.4 | 1.011x |
| mxfp8_gated | `dswiglu` | col | 8192x28672 | Llama-3-70B FFN | 460.1 | 422.2 | 1.090x | 458.1 | 420.6 | 1.089x |
| mxfp8_gated | `dswiglu` | col | 16384x5120 | long-seq hidden | 168.8 | 155.3 | 1.087x | 167.2 | 153.5 | 1.089x |
| mxfp8_gated | `dswiglu` | col | 8192x8192 | Llama-3-70B hidden | 136.8 | 126.0 | 1.086x | 135.3 | 124.4 | 1.087x |
| mxfp8_gated | `dswiglu` | col | 4096x14336 | Llama-3-8B FFN | 120.8 | 111.4 | 1.084x | 119.4 | 109.6 | 1.089x |
| mxfp8_gated | `dswiglu` | col | 2048x12288 | GPT-3-175B hidden | 56.4 | 52.1 | 1.082x | 54.8 | 50.7 | 1.081x |
| mxfp8_gated | `dswiglu` | col | 4096x4096 | Llama-3-8B hidden | 40.3 | 37.3 | 1.079x | 38.7 | 35.7 | 1.081x |
| mxfp8_gated | `dswiglu` | row | 16384x5120 | long-seq hidden | 201.7 | 198.1 | 1.018x | 199.3 | 195.9 | 1.017x |
| mxfp8_gated | `dswiglu` | row | 4096x4096 | Llama-3-8B hidden | 47.7 | 46.9 | 1.018x | 45.9 | 44.8 | 1.024x |
| mxfp8_gated | `dswiglu` | row | 8192x8192 | Llama-3-70B hidden | 163.2 | 160.8 | 1.015x | 161.3 | 158.9 | 1.015x |
| mxfp8_gated | `dswiglu` | row | 2048x12288 | GPT-3-175B hidden | 67.0 | 66.1 | 1.014x | 64.6 | 63.3 | 1.021x |
| mxfp8_gated | `dswiglu` | row | 4096x14336 | Llama-3-8B FFN | 144.2 | 142.2 | 1.014x | 141.8 | 140.0 | 1.013x |
| mxfp8_gated | `dswiglu` | row | 8192x28672 | Llama-3-70B FFN | 547.6 | 541.0 | 1.012x | 545.2 | 539.4 | 1.011x |
| mxfp8_gated | `geglu` | both | 4096x14336 | Llama-3-8B FFN | 122.0 | 114.7 | 1.064x | 121.2 | 114.2 | 1.062x |
| mxfp8_gated | `geglu` | both | 8192x28672 | Llama-3-70B FFN | 473.5 | 445.3 | 1.063x | 471.4 | 444.8 | 1.060x |
| mxfp8_gated | `geglu` | both | 8192x8192 | Llama-3-70B hidden | 138.9 | 130.9 | 1.061x | 137.7 | 129.5 | 1.064x |
| mxfp8_gated | `geglu` | both | 16384x5120 | long-seq hidden | 171.7 | 161.8 | 1.061x | 171.2 | 161.0 | 1.063x |
| mxfp8_gated | `geglu` | both | 2048x12288 | GPT-3-175B hidden | 55.5 | 52.6 | 1.056x | 54.7 | 51.8 | 1.057x |
| mxfp8_gated | `geglu` | both | 4096x4096 | Llama-3-8B hidden | 39.3 | 37.2 | 1.055x | 37.6 | 35.6 | 1.057x |
| mxfp8_gated | `geglu` | col | 2048x12288 | GPT-3-175B hidden | 35.9 | 35.6 | 1.009x | 35.3 | 35.0 | 1.009x |
| mxfp8_gated | `geglu` | col | 4096x14336 | Llama-3-8B FFN | 77.0 | 76.5 | 1.007x | 76.0 | 75.5 | 1.006x |
| mxfp8_gated | `geglu` | col | 16384x5120 | long-seq hidden | 107.6 | 106.9 | 1.006x | 106.1 | 105.7 | 1.003x |
| mxfp8_gated | `geglu` | col | 4096x4096 | Llama-3-8B hidden | 25.6 | 25.4 | 1.006x | 23.6 | 23.6 | 1.000x |
| mxfp8_gated | `geglu` | col | 8192x8192 | Llama-3-70B hidden | 87.1 | 86.7 | 1.005x | 86.1 | 85.6 | 1.006x |
| mxfp8_gated | `geglu` | col | 8192x28672 | Llama-3-70B FFN | 289.2 | 288.6 | 1.002x | 288.0 | 287.5 | 1.002x |
| mxfp8_gated | `geglu` | row | 2048x12288 | GPT-3-175B hidden | 37.0 | 34.1 | 1.085x | 36.0 | 33.0 | 1.089x |
| mxfp8_gated | `geglu` | row | 8192x28672 | Llama-3-70B FFN | 285.6 | 264.4 | 1.080x | 284.1 | 261.8 | 1.085x |
| mxfp8_gated | `geglu` | row | 4096x4096 | Llama-3-8B hidden | 26.6 | 24.7 | 1.078x | 24.1 | 22.5 | 1.073x |
| mxfp8_gated | `geglu` | row | 4096x14336 | Llama-3-8B FFN | 77.2 | 71.7 | 1.076x | 75.9 | 69.8 | 1.087x |
| mxfp8_gated | `geglu` | row | 16384x5120 | long-seq hidden | 107.1 | 99.6 | 1.075x | 105.4 | 97.0 | 1.087x |
| mxfp8_gated | `geglu` | row | 8192x8192 | Llama-3-70B hidden | 87.1 | 81.2 | 1.072x | 85.7 | 79.2 | 1.082x |
| mxfp8_gated | `swiglu` | both | 4096x4096 | Llama-3-8B hidden | 43.3 | 40.9 | 1.060x | 40.5 | 38.8 | 1.045x |
| mxfp8_gated | `swiglu` | both | 2048x12288 | GPT-3-175B hidden | 61.3 | 58.9 | 1.040x | 59.9 | 57.8 | 1.037x |
| mxfp8_gated | `swiglu` | both | 8192x8192 | Llama-3-70B hidden | 149.0 | 147.1 | 1.013x | 146.6 | 146.2 | 1.003x |
| mxfp8_gated | `swiglu` | both | 8192x28672 | Llama-3-70B FFN | 502.7 | 500.4 | 1.004x | 499.4 | 499.7 | 0.999x |
| mxfp8_gated | `swiglu` | both | 4096x14336 | Llama-3-8B FFN | 130.1 | 130.0 | 1.001x | 128.4 | 128.4 | 1.001x |
| mxfp8_gated | `swiglu` | both | 16384x5120 | long-seq hidden | 182.9 | 182.9 | 1.000x | 181.0 | 180.4 | 1.003x |
| mxfp8_gated | `swiglu` | col | 2048x12288 | GPT-3-175B hidden | 37.9 | 37.2 | 1.016x | 37.3 | 36.9 | 1.013x |
| mxfp8_gated | `swiglu` | col | 4096x4096 | Llama-3-8B hidden | 27.0 | 26.5 | 1.016x | 25.0 | 24.4 | 1.025x |
| mxfp8_gated | `swiglu` | col | 16384x5120 | long-seq hidden | 113.4 | 111.7 | 1.015x | 112.7 | 110.9 | 1.016x |
| mxfp8_gated | `swiglu` | col | 4096x14336 | Llama-3-8B FFN | 81.0 | 79.9 | 1.014x | 80.3 | 79.2 | 1.015x |
| mxfp8_gated | `swiglu` | col | 8192x28672 | Llama-3-70B FFN | 307.0 | 303.0 | 1.013x | 306.3 | 302.1 | 1.014x |
| mxfp8_gated | `swiglu` | col | 8192x8192 | Llama-3-70B hidden | 91.9 | 90.7 | 1.013x | 91.1 | 89.8 | 1.015x |
| mxfp8_gated | `swiglu` | row | 8192x28672 | Llama-3-70B FFN | 338.4 | 333.1 | 1.016x | 336.4 | 330.3 | 1.019x |
| mxfp8_gated | `swiglu` | row | 16384x5120 | long-seq hidden | 125.9 | 124.5 | 1.011x | 124.1 | 122.2 | 1.016x |
| mxfp8_gated | `swiglu` | row | 8192x8192 | Llama-3-70B hidden | 102.4 | 101.7 | 1.007x | 100.5 | 99.2 | 1.013x |
| mxfp8_gated | `swiglu` | row | 4096x14336 | Llama-3-8B FFN | 90.6 | 90.2 | 1.005x | 89.0 | 87.8 | 1.014x |
| mxfp8_gated | `swiglu` | row | 2048x12288 | GPT-3-175B hidden | 43.5 | 43.7 | 0.994x | 42.3 | 41.8 | 1.011x |
| mxfp8_gated | `swiglu` | row | 4096x4096 | Llama-3-8B hidden | 30.9 | 31.3 | 0.987x | 26.5 | 26.2 | 1.010x |
| nvfp4 | `nvfp42d` | both | 2048x12288 | GPT-3-175B hidden | 22.0 | 21.1 | 1.039x | 21.3 | 20.4 | 1.040x |
| nvfp4 | `nvfp42d` | both | 4096x4096 | Llama-3-8B hidden | 15.4 | 14.8 | 1.037x | 15.2 | 14.6 | 1.043x |
| nvfp4 | `nvfp42d` | both | 4096x14336 | Llama-3-8B FFN | 45.5 | 44.4 | 1.026x | 45.1 | 44.5 | 1.014x |
| nvfp4 | `nvfp42d` | both | 16384x5120 | long-seq hidden | 65.3 | 63.8 | 1.024x | 64.7 | 63.9 | 1.013x |
| nvfp4 | `nvfp42d` | both | 8192x8192 | Llama-3-70B hidden | 51.2 | 50.1 | 1.023x | 51.1 | 49.9 | 1.023x |
| nvfp4 | `nvfp42d` | both | 8192x28672 | Llama-3-70B FFN | 188.7 | 188.0 | 1.004x | 188.8 | 187.9 | 1.005x |
| nvfp4 | `nvfp42d` | col | 4096x4096 | Llama-3-8B hidden | 11.8 | 11.0 | 1.068x | 11.4 | 10.8 | 1.058x |
| nvfp4 | `nvfp42d` | col | 2048x12288 | GPT-3-175B hidden | 16.4 | 15.6 | 1.053x | 15.8 | 14.9 | 1.060x |
| nvfp4 | `nvfp42d` | col | 16384x5120 | long-seq hidden | 48.9 | 47.6 | 1.026x | 48.6 | 47.4 | 1.026x |
| nvfp4 | `nvfp42d` | col | 8192x28672 | Llama-3-70B FFN | 149.9 | 147.2 | 1.018x | 149.6 | 147.0 | 1.017x |
| nvfp4 | `nvfp42d` | col | 4096x14336 | Llama-3-8B FFN | 33.7 | 33.2 | 1.015x | 33.4 | 32.9 | 1.014x |
| nvfp4 | `nvfp42d` | col | 8192x8192 | Llama-3-70B hidden | 40.3 | 39.7 | 1.014x | 40.3 | 39.4 | 1.021x |
| nvfp4 | `nvfp42d` | row | 8192x28672 | Llama-3-70B FFN | 96.2 | 93.3 | 1.031x | 97.0 | 94.1 | 1.030x |
| nvfp4 | `nvfp42d` | row | 2048x12288 | GPT-3-175B hidden | 14.7 | 14.3 | 1.030x | 13.9 | 13.6 | 1.024x |
| nvfp4 | `nvfp42d` | row | 16384x5120 | long-seq hidden | 37.7 | 36.6 | 1.030x | 37.6 | 36.5 | 1.030x |
| nvfp4 | `nvfp42d` | row | 8192x8192 | Llama-3-70B hidden | 31.7 | 30.8 | 1.029x | 31.4 | 30.9 | 1.016x |
| nvfp4 | `nvfp42d` | row | 4096x14336 | Llama-3-8B FFN | 28.7 | 28.0 | 1.024x | 28.0 | 27.2 | 1.028x |
| nvfp4 | `nvfp42d` | row | 4096x4096 | Llama-3-8B hidden | 10.3 | 10.1 | 1.022x | 9.9 | 9.7 | 1.023x |
| specialized | `plain` | both | 2048x12288 | GPT-3-175B hidden | 20.3 | 20.0 | 1.014x | 16.2 | 14.9 | 1.087x |
| specialized | `plain` | both | 8192x8192 | Llama-3-70B hidden | 47.1 | 46.8 | 1.006x | 44.2 | 44.0 | 1.006x |
| specialized | `plain` | both | 16384x5120 | long-seq hidden | 57.5 | 57.2 | 1.005x | 54.3 | 54.0 | 1.005x |
| specialized | `plain` | both | 8192x28672 | Llama-3-70B FFN | 149.6 | 149.4 | 1.001x | 145.9 | 146.0 | 0.999x |
| specialized | `plain` | both | 4096x14336 | Llama-3-8B FFN | 42.1 | 42.2 | 0.998x | 39.8 | 39.5 | 1.007x |
| specialized | `plain` | both | 4096x4096 | Llama-3-8B hidden | 16.0 | 16.1 | 0.997x | 11.8 | 10.4 | 1.132x |
| specialized | `plain` | row | 16384x5120 | long-seq hidden | 38.6 | 38.5 | 1.002x | 36.7 | 36.7 | 1.000x |
| specialized | `plain` | row | 4096x14336 | Llama-3-8B FFN | 28.2 | 28.1 | 1.001x | 25.1 | 25.1 | 0.999x |
| specialized | `plain` | row | 4096x4096 | Llama-3-8B hidden | 10.0 | 10.0 | 1.000x | 7.3 | 7.4 | 0.996x |
| specialized | `plain` | row | 8192x28672 | Llama-3-70B FFN | 101.0 | 101.1 | 1.000x | 98.8 | 98.6 | 1.002x |
| specialized | `plain` | row | 2048x12288 | GPT-3-175B hidden | 13.9 | 13.9 | 0.999x | 10.2 | 10.1 | 1.008x |
| specialized | `plain` | row | 8192x8192 | Llama-3-70B hidden | 31.3 | 31.3 | 0.998x | 28.6 | 28.6 | 1.000x |

</details>

## Patched but not measurable

Three further sites carry the identical anti-pattern but cannot be reached, so they are fixed for consistency rather than measured -- leaving them would let the pattern be reintroduced if those paths are ever enabled.

| site | why unreachable |
|---|---|
| `cast/mxfp8/specialized/quantize_mxfp8.cuh` (warp-specialized variant) | guarded by `enable_if_t<CastTraits::_use_warp_specialization>`, and that flag is `false`; the library contains only the 8 non-warp-specialized instantiations |
| `cast/nvfp4/quantize_transpose_nvfp4.cuh` (1D kernel) | `quantize_transpose<false>` is only called for bf16, and its first statement diverts every bf16 1D call to `quantize_transpose_tuned_1D` |
| `cast/nvfp4/group_quantize_transpose_nvfp4.cuh` | the PyTorch grouped path requires RHT (`"graph safe grouped quant kernel for non-RHT path is not ready yet"`), which routes to the hadamard fusion kernels instead |

`hadamard_transform/*.cu` carries the same pattern at 3 more sites but was deliberately left alone: across all 108 hadamard kernels there are only 72 generic `LD`/`ST` in total (worst kernel: 6), so there is nothing measurable to win.

## Method

- Both arms are full builds of the same tree, profiled back to back **on one node**. Each profile records the hostname and the md5 of the `libtransformer_engine.so` actually mapped into the process (read from `/proc/self/maps` after load), so neither a stale build nor a cross-node comparison can slip through unnoticed.
- **cold L2**: 512 MB scratch zeroed before every iteration. **warm L2**: no flush, back-to-back steady state. Both are pure kernel time; the cache state only changes what the kernel is bottlenecked on.
- Numerics are **bitwise identical** across the two builds (90 configurations spanning MXFP8 plain/gelu/dgelu/dbias/swiglu/geglu, NVFP4 1D and 2D, and FP8 swiglu/geglu). This is a pure codegen change.
- Gains track how scalar the shared-memory access pattern is, because the generic-address penalty is charged per instruction: paths moving ~1.5-1.7 bytes/instruction gain most, while already-vectorized paths (~10 bytes/instruction) barely move.
- Reproduce with `./run_final.sh && python gen_final_report.py`.
