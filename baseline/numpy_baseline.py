import numpy as np
import time
import json 


cpu_calc = list() 
n_exec = 10
dtype = np.float16
mat_sizes = [256, 512, 1024, 2048, 3072, 4096, 8192]
for shape in mat_sizes:
    print(f"\nCalculating {shape} matmul ... ")
    in0 = np.ones((shape, shape), dtype=dtype)
    in1 = np.random.random((shape, shape)).astype(dtype)
    start = time.time()
    for _ in range(n_exec):
        np.matmul(in0, in1)
    tot_time = (time.time() - start) * 1e6
    tot_time /= n_exec
    print(f"Avg time: {tot_time}")
    cpu_calc.append(tot_time)
cpu_calc

with open("./results/np_baseline.json", "w") as f:
    json.dump(cpu_calc, f)    