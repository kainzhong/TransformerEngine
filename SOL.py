import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=32)
parser.add_argument("-T", type=int, default=4096)
parser.add_argument("-C", type=int, default=4096)
parser.add_argument("-n", type=int, default=4)
args = parser.parse_args()

B = args.B
T = args.T
C = args.C
n = args.n
dtype = "float32"
bytes_per_elem = 4
GPU = "B200"

hardware_specs = {
    "B200": {
        "MEM_BANDWIDTH": 8.0e12,  # 8 TB/s
        "FP32_FLOPS": 75.0e12,    # 75 TFLOPS (Vector FP32 for tiny inner dims)
        "TF32_FLOPS": 1.125e15,   # 1,125 TFLOPS (Dense TF32 Tensor Core)
    }
}

print(f"================================================================================")
print(f" GPU PERFORMANCE ESTIMATOR ({GPU})")
print(f"================================================================================")
print(f" Config: B={B}, T={T}, C={C}, n={n}")
print(f" Format: {dtype} ({bytes_per_elem} bytes/elem)")
print(f"================================================================================\n")

def print_sol_breakdown(name, mem_gb, flops_g=None, use_tf32=False):
    bw_gb_s = hardware_specs[GPU]["MEM_BANDWIDTH"] / 1e9
    
    # Select peak FLOPs based on TensorCore utilization
    if use_tf32:
        peak_flops_g = hardware_specs[GPU]["TF32_FLOPS"] / 1e9
        math_type = "TF32 TensorCore"
    else:
        peak_flops_g = hardware_specs[GPU]["FP32_FLOPS"] / 1e9
        math_type = "FP32 Vector"

    time_mem_ms = (mem_gb / bw_gb_s) * 1000
    
    print(f"[{name}]")
    if flops_g is not None:
        time_math_ms = (flops_g / peak_flops_g) * 1000
        sol_time = max(time_mem_ms, time_math_ms)
        bound = "FLOPS bounded" if time_math_ms > time_mem_ms else "Memory bounded"
        
        print(f"  ├─ Architecture : {math_type}")
        print(f"  ├─ Total Mem R/W: {mem_gb:8.4f} GB")
        print(f"  ├─ Total Math   : {flops_g:8.4f} GFLOPS")
        print(f"  ├─ Mem Time     : {time_mem_ms:8.4f} ms")
        print(f"  ├─ Math Time    : {time_math_ms:8.4f} ms")
        print(f"  └─ SOL Time     : {sol_time:8.4f} ms ({bound})\n")
    else:
        sol_time = time_mem_ms
        bound = "Memory bounded"
        print(f"  ├─ Total Mem R/W: {mem_gb:8.4f} GB")
        print(f"  ├─ Mem Time     : {time_mem_ms:8.4f} ms")
        print(f"  └─ SOL Time     : {sol_time:8.4f} ms ({bound})\n")

# ---------------------------------------------------------
# 1. Projection kernel: (B, T, n*C) @ (n*C, 32)
# ---------------------------------------------------------
in1_gb = B * T * (n * C) * bytes_per_elem / 1e9
in2_gb = (n * C) * 32 * bytes_per_elem / 1e9
out_gb = B * T * 32 * bytes_per_elem / 1e9
proj_mem_gb = in1_gb + in2_gb + out_gb
proj_flops_g = 2 * (B * T) * 32 * (n * C) / 1e9

print(f"================================================================================")
print(f"1. Projection Kernel: (B, T, n*C) @ (n*C, 32)")
print(f"   Input 1: {B} * {T} * ({n} * {C}) * {bytes_per_elem} / 1e9 = {in1_gb:.4f} GB")
print(f"   Input 2: ({n} * {C}) * 32 * {bytes_per_elem} / 1e9 = {in2_gb:.4f} GB")
print(f"   Output : {B} * {T} * 32 * {bytes_per_elem} / 1e9 = {out_gb:.4f} GB")
print(f"   FLOPs  : 2 * ({B} * {T}) * 32 * ({n} * {C}) / 1e9 = {proj_flops_g:.4f} GFLOPS")
print_sol_breakdown("Projection Summary", proj_mem_gb, proj_flops_g, use_tf32=True)


# ---------------------------------------------------------
# 2. Element-wise kernel: (B, T, 32)
# ---------------------------------------------------------
ew_read_gb = B * T * 32 * bytes_per_elem / 1e9
ew_write_gb = ew_read_gb
ew_mem_gb = ew_read_gb + ew_write_gb

print(f"================================================================================")
print(f"2. Element-wise Kernel: (B, T, 32)")
print(f"   Read   : {B} * {T} * 32 * {bytes_per_elem} / 1e9 = {ew_read_gb:.4f} GB")
print(f"   Write  : {B} * {T} * 32 * {bytes_per_elem} / 1e9 = {ew_write_gb:.4f} GB")
print_sol_breakdown("Element-wise Summary", ew_mem_gb)


# ---------------------------------------------------------
# 3. Sinkhorn kernel: (B, T, n, n)
# ---------------------------------------------------------
sink_read_gb = B * T * n * n * bytes_per_elem / 1e9
sink_write_gb = sink_read_gb
sink_mem_gb = sink_read_gb + sink_write_gb

print(f"================================================================================")
print(f"3. Sinkhorn Kernel: (B, T, n, n)")
print(f"   Read   : {B} * {T} * {n} * {n} * {bytes_per_elem} / 1e9 = {sink_read_gb:.6f} GB")
print(f"   Write  : {B} * {T} * {n} * {n} * {bytes_per_elem} / 1e9 = {sink_write_gb:.6f} GB")
print_sol_breakdown("Sinkhorn Summary", sink_mem_gb)


# ---------------------------------------------------------
# 4. Pre kernel: (B, T, 1, n) @ (B, T, n, C) batched
# ---------------------------------------------------------
pre_in1_gb = B * T * 1 * n * bytes_per_elem / 1e9
pre_in2_gb = B * T * n * C * bytes_per_elem / 1e9
pre_out_gb = B * T * 1 * C * bytes_per_elem / 1e9
pre_mem_gb = pre_in1_gb + pre_in2_gb + pre_out_gb
pre_flops_g = B * T * (2 * 1 * n * C) / 1e9

print(f"================================================================================")
print(f"4. Pre Kernel: (B, T, 1, n) @ (B, T, n, C)")
print(f"   Input 1: {B} * {T} * 1 * {n} * {bytes_per_elem} / 1e9 = {pre_in1_gb:.6f} GB")
print(f"   Input 2: {B} * {T} * {n} * {C} * {bytes_per_elem} / 1e9 = {pre_in2_gb:.6f} GB")
print(f"   Output : {B} * {T} * 1 * {C} * {bytes_per_elem} / 1e9 = {pre_out_gb:.6f} GB")
print(f"   FLOPs  : {B} * {T} * (2 * 1 * {n} * {C}) / 1e9 = {pre_flops_g:.6f} GFLOPS")
print_sol_breakdown("Pre Summary", pre_mem_gb, pre_flops_g, use_tf32=False)


# ---------------------------------------------------------
# 5. Post + Res kernel (Fused)
# ---------------------------------------------------------
post_in1_1_gb = B * T * n * 1 * bytes_per_elem / 1e9
post_in1_2_gb = B * T * 1 * C * bytes_per_elem / 1e9
post_in2_1_gb = B * T * n * n * bytes_per_elem / 1e9
post_in2_2_gb = B * T * n * C * bytes_per_elem / 1e9
post_out_gb   = B * T * n * C * bytes_per_elem / 1e9

post_mem_gb = post_in1_1_gb + post_in1_2_gb + post_in2_1_gb + post_in2_2_gb + post_out_gb

flops_term1_g = B * T * (2 * n * 1 * C) / 1e9
flops_term2_g = B * T * (2 * n * n * C) / 1e9
flops_add_g   = B * T * n * C / 1e9
post_flops_g  = flops_term1_g + flops_term2_g + flops_add_g

print(f"================================================================================")
print(f"5. Post + Res Kernel (Fused): (B, T, n, 1) @ (B, T, 1, C) + (B, T, n, n) @ (B, T, n, C)")
print(f"   In 1.1 : {B} * {T} * {n} * 1 * {bytes_per_elem} / 1e9 = {post_in1_1_gb:.6f} GB")
print(f"   In 1.2 : {B} * {T} * 1 * {C} * {bytes_per_elem} / 1e9 = {post_in1_2_gb:.6f} GB")
print(f"   In 2.1 : {B} * {T} * {n} * {n} * {bytes_per_elem} / 1e9 = {post_in2_1_gb:.6f} GB")
print(f"   In 2.2 : {B} * {T} * {n} * {C} * {bytes_per_elem} / 1e9 = {post_in2_2_gb:.6f} GB")
print(f"   Output : {B} * {T} * {n} * {C} * {bytes_per_elem} / 1e9 = {post_out_gb:.6f} GB")
print(f"   FLOPs 1: {B} * {T} * (2 * {n} * 1 * {C}) / 1e9 = {flops_term1_g:.6f} GFLOPS")
print(f"   FLOPs 2: {B} * {T} * (2 * {n} * {n} * {C}) / 1e9 = {flops_term2_g:.6f} GFLOPS")
print(f"   FLOP Add: {B} * {T} * {n} * {C} / 1e9 = {flops_add_g:.6f} GFLOPS")
print_sol_breakdown("Post + Res Summary", post_mem_gb, post_flops_g, use_tf32=False)
print(f"================================================================================")
