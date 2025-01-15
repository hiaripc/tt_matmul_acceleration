import torch
import time
import json 


cpu_calc = list() 
n_exec = 1
dtype = torch.bfloat16
# 256, 512, 1024, 2048, 3072, 4096 
mat_sizes = [8192]
for shape in mat_sizes:
    print(f"\nCalculating {shape} matmul ... ")
    in0 = torch.ones((shape, shape), dtype=dtype)
    in1 = torch.randn((shape, shape), dtype=dtype)
    start = time.time()
    for _ in range(n_exec):
        torch.matmul(in0, in1)
    tot_time = (time.time() - start)
    tot_time /= n_exec
    tot_time *= 1e6
    print(f"Avg time: {tot_time}")
    cpu_calc.append(tot_time)
cpu_calc

with open("./torch_baseline.json", "w") as f:
    json.dump(cpu_calc, f)    