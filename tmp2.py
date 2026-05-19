import time

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

import transformer_engine_torch as tex
import torch

rowwise = True
colwise = False
M = 4096
N = 4096

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=rowwise,
    columnwise=colwise,
)
quantizer.internal = True

# warmup
for i in range(1000):
    rowwise_data = torch.empty((M, N), dtype=torch.uint8, device="cuda")
    rowwise_scale_inv = torch.empty((M, N // 32), dtype=torch.uint8, device="cuda")

t0 = time.perf_counter_ns()
for i in range(5000):
    rowwise_data = torch.empty((M, N), dtype=torch.uint8, device="cuda")
    rowwise_scale_inv = torch.empty((M, N // 32), dtype=torch.uint8, device="cuda")
t1 = time.perf_counter_ns()
avg_time = (t1 - t0) / 5000 / 1000
print(f"Average time: {avg_time:.6f} us")