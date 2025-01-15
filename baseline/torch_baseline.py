import torch
import time
import csv

test_name = "torch_fp16"

with open(f"../results/{test_name}.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["m", test_name])
    n_exec = 100
    dtype = torch.float32
    mat_sizes = [256, 512, 1024, 2048, 3072, 4096, 8192 ]
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
        writer.writerow([shape, tot_time])