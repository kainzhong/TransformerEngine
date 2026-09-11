"""Summarize quantize_bidim_kernel time from an nsys sqlite export.

Kernel name alone cannot separate the shapes: K_COMPILE_TIME is a template
parameter, so 16384x7168 and 32768x7168 instantiate the same kernel.  Group by
(name, gridX, gridY) instead -- the grid is what identifies the shape -- and
keep only the last `iters` launches of each group, which drops the warmup.
"""

import argparse
import sqlite3
import statistics

p = argparse.ArgumentParser()
p.add_argument("sqlite")
p.add_argument("--iters", type=int, default=100)
args = p.parse_args()

con = sqlite3.connect(args.sqlite)
rows = con.execute("""
    SELECT s.value, k.gridX, k.gridY, k.blockX, k.end - k.start, k.start
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON s.id = k.demangledName
    WHERE s.value LIKE '%quantize_bidim_kernel%'
    ORDER BY k.start
""").fetchall()

groups = {}
for name, gx, gy, bx, dur, start in rows:
    groups.setdefault((gx, gy), {"name": name, "block": bx, "dur": []})["dur"].append(dur)

hdr = (f"{'shape':>14}  {'grid':>10}  {'n':>4}  {'mean us':>8}  {'med us':>8}  "
       f"{'min us':>8}  {'GB/s':>7}  {'%peak':>6}")
print(hdr)
print("-" * len(hdr))

PEAK = 8000.0  # GB200 HBM3e, GB/s

for (gx, gy), g in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    d = g["dur"][-args.iters:]
    M, N = gy * 32, gx * 256
    # in: bf16.  out: rowwise fp8 + colwise fp8 + both e8m0 scale planes.
    nbytes = 2 * M * N + 2 * M * N + 2 * (M * N / 32)
    mean = statistics.mean(d) / 1e3
    med = statistics.median(d) / 1e3
    gbs = nbytes / (mean * 1e3)
    print(f"{M}x{N:<8}  {gx}x{gy:<7}  {len(d):>4}  {mean:8.2f}  {med:8.2f}  "
          f"{min(d)/1e3:8.2f}  {gbs:7.1f}  {100*gbs/PEAK:5.1f}%")

print()
for (gx, gy), g in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    print(f"  {gy*32}x{gx*256}: {g['name']}  block={g['block']}  "
          f"launches={len(g['dur'])}")
