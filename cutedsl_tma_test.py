"""Benchmark host-side TMA descriptor creation overhead.

Each subprocess runs an empty kernel that receives `n` TMA atoms created on
the host (matching the pattern in `quantize_mxfp8_cutedsl_alt.py`'s
`MXFP8QuantizeSmemKernel.__call__` — `cute.nvgpu.cpasync.make_tiled_tma_atom`
on a 2-D bf16 gmem tensor). The slope of latency vs. `n` is the marginal
host-side cost of creating one extra TMA descriptor per invocation.
"""

import os
import subprocess
import sys
import tempfile


SCRIPT_TEMPLATE = """\
import time
import torch

import cutlass
import cutlass.cute as cute

TILE_Y = 32
TILE_X = 64

{fake_decls}
{tensor_decls}

class TmaKernel:

    @cute.jit
    def __call__(self, {x_params}):
        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_tile_layout = cute.make_ordered_layout((TILE_Y, TILE_X), order=(1, 0))
        cta_tiler = (TILE_Y, TILE_X)
{atom_decls}
        self.kernel({kernel_args}).launch(
            grid=(1,),
            block=(1,),
        )

    @cute.kernel
    def kernel(self, {kernel_params}):
        return


WARMUP = 100
TRIALS = 10
ITERS = 400

compiled = cute.compile(TmaKernel(), {x_fakes}options="--enable-tvm-ffi")

# Warmup
for _ in range(WARMUP):
    compiled({x_args})
torch.cuda.synchronize()

trial_us = []
for _ in range(TRIALS):
    t0 = time.perf_counter_ns()
    for _ in range(ITERS):
        compiled({x_args})
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    trial_us.append((t1 - t0) / ITERS / 1000)

mean = sum(trial_us) / len(trial_us)
var = sum((x - mean) ** 2 for x in trial_us) / (len(trial_us) - 1)
std = var ** 0.5
print(f'RESULT TmaKernel {{mean}} {{std}}', flush=True)
"""


def bench_n_tma(n: int):
    x_params = ", ".join([f"x{i}" for i in range(n)])
    x_fakes = "".join([f"x_fake{i}, " for i in range(n)])
    x_args = ", ".join([f"x{i}" for i in range(n)])
    fake_decls = "\n".join(
        f"x_fake{i} = cute.runtime.make_fake_compact_tensor("
        f"cute.BFloat16, (4096, 4096), stride_order=(1, 0), "
        f"memspace=cute.AddressSpace.gmem, assumed_align=16)"
        for i in range(n)
    )
    tensor_decls = "\n".join(
        f"x{i} = torch.empty((4096, 4096), dtype=torch.bfloat16, device='cuda')"
        for i in range(n)
    )
    atom_decls = "\n".join(
        f"        atom{i}, src{i} = cute.nvgpu.cpasync.make_tiled_tma_atom("
        f"op_load, x{i}, smem_tile_layout, cta_tiler, num_multicast=1)"
        for i in range(n)
    )
    kernel_args = ", ".join([f"atom{i}, src{i}" for i in range(n)])
    kernel_params = ", ".join([f"atom{i}, src{i}" for i in range(n)])

    script = SCRIPT_TEMPLATE.format(
        x_params=x_params,
        x_fakes=x_fakes,
        x_args=x_args,
        fake_decls=fake_decls,
        tensor_decls=tensor_decls,
        atom_decls=atom_decls,
        kernel_args=kernel_args,
        kernel_params=kernel_params,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path], check=True, capture_output=True, text=True
        )
    finally:
        os.unlink(path)

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            _, _, mean, std = line.split()
            return float(mean), float(std)
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    return None


rows = []
for i in range(10):
    print(f"Benchmarking {i} TMA atoms...", flush=True)
    r = bench_n_tma(i)
    rows.append((i, r))

print()
header = f"{'n_tma':>6} | {'latency mean±σ (us)':>22}"
print(header)
print("-" * len(header))
for n, r in rows:
    if r is None:
        print(f"{n:>6} | {'n/a':>22}")
    else:
        print(f"{n:>6} |  {r[0]:8.3f} ± {r[1]:6.3f}")


def linreg(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx
    intercept = my - slope * mx
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else float("nan")
    return slope, intercept, r2


xs = [n for n, r in rows if r is not None]
ys = [r[0] for n, r in rows if r is not None]
if len(xs) >= 2:
    slope, intercept, r2 = linreg(xs, ys)
    print()
    print("Linear fit  latency(n) = intercept + slope * n")
    print(
        f"  slope={slope:7.3f} us/tma_atom  intercept={intercept:7.3f} us  R^2={r2:.4f}"
    )

