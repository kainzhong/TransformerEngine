import os
import random
import subprocess
import sys
import tempfile


SCRIPT_TEMPLATE = """\
import time
import torch

import cutlass
import cutlass.cute as cute

{fake_decls}
{tensor_decls}

class NoLaunchKernel:

    @cute.jit
    def __call__(self, {x_params}):
        return

class LaunchKernel:

    @cute.jit
    def __call__(self, {x_params}):
        self.kernel({x_params}).launch(
            grid=(1,),
            block=(1,),
        )

    @cute.kernel
    def kernel(self, {x_params}):
        return

WARMUP = 100
TRIALS = 10
ITERS = 400

def bench(kernel_cls, with_alloc):
    compiled = cute.compile(kernel_cls(), {x_fakes}options="--enable-tvm-ffi")

    # Warmup (include the alloc path so caching allocator is hot)
    for i in range(WARMUP):
        if with_alloc:
            torch.empty(1, dtype=torch.uint8, device='cuda')
        compiled({x_args})
    torch.cuda.synchronize()

    trial_us = []
    for _ in range(TRIALS):
        t0 = time.perf_counter_ns()
        if with_alloc:
            for i in range(ITERS):
                torch.empty(1, dtype=torch.uint8, device='cuda')
                compiled({x_args})
        else:
            for i in range(ITERS):
                compiled({x_args})
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        trial_us.append((t1 - t0) / ITERS / 1000)

    mean = sum(trial_us) / len(trial_us)
    var = sum((x - mean) ** 2 for x in trial_us) / (len(trial_us) - 1)
    std = var ** 0.5
    suffix = '+Alloc' if with_alloc else ''
    print(f'RESULT {{kernel_cls.__name__}}{{suffix}} {{mean}} {{std}}', flush=True)

bench(NoLaunchKernel, False)
bench(LaunchKernel,   False)
bench(NoLaunchKernel, True)
bench(LaunchKernel,   True)
"""


SHAPE_CHOICES = [128, 256, 512, 1024, 2048, 4096]


def bench_n_args(n: int):
    rng = random.Random(0xC0FFEE + n)
    shapes = [
        (rng.choice(SHAPE_CHOICES), rng.choice(SHAPE_CHOICES)) for _ in range(n)
    ]
    x_params = ", ".join([f"x{i}" for i in range(n)])
    x_fakes = "".join([f"x_fake{i}, " for i in range(n)])
    x_args = ", ".join([f"x{i}" for i in range(n)])
    fake_decls = "\n".join(
        f"x_fake{i} = cute.runtime.make_fake_compact_tensor(cute.BFloat16, {shapes[i]}, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=16)"
        for i in range(n)
    )
    tensor_decls = "\n".join(
        f"x{i} = torch.empty({shapes[i]}, dtype=torch.bfloat16, device='cuda')"
        for i in range(n)
    )

    script = SCRIPT_TEMPLATE.format(
        x_params=x_params,
        x_fakes=x_fakes,
        x_args=x_args,
        fake_decls=fake_decls,
        tensor_decls=tensor_decls,
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

    results = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            _, name, mean, std = line.split()
            results[name] = (float(mean), float(std))
    return results


rows = []
for i in range(10):
    print(f"Benchmarking {i} args...", flush=True)
    r = bench_n_args(i)
    rows.append((
        i,
        r.get("NoLaunchKernel"),
        r.get("LaunchKernel"),
        r.get("NoLaunchKernel+Alloc"),
        r.get("LaunchKernel+Alloc"),
    ))

print()
columns = [
    ("NoLaunch (us)",         1),
    ("Launch (us)",           2),
    ("NoLaunch+Alloc (us)",   3),
    ("Launch+Alloc (us)",     4),
]
header = f"{'n_args':>6} | " + " | ".join(f"{c[0]:>22}" for c in columns)
print(header)
print("-" * len(header))
for row in rows:
    cells = [f"{row[0]:>6}"]
    for _, idx in columns:
        v = row[idx]
        s = f"{v[0]:8.3f} ± {v[1]:6.3f}" if v else "n/a"
        cells.append(f"{s:>22}")
    print(" | ".join(cells))


def linreg(xs, ys):
    """Ordinary least-squares slope/intercept and R^2."""
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


print()
print("Linear fit  latency(n) = intercept + slope * n")
for name, idx in [("NoLaunch", 1), ("Launch", 2),
                  ("NoLaunch+Alloc", 3), ("Launch+Alloc", 4)]:
    xs = [r[0] for r in rows if r[idx] is not None]
    ys = [r[idx][0] for r in rows if r[idx] is not None]
    if len(xs) >= 2:
        slope, intercept, r2 = linreg(xs, ys)
        print(
            f"  {name:>16}: slope={slope:7.3f} us/param  intercept={intercept:7.3f} us  R^2={r2:.4f}"
        )

