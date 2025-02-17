import torch
import time
import csv
import os

dir_test = "../results/"
test_name = "torch_fp32_cores"
core_selections = [1, 2, 4, 8]
# mat_sizes = [256, 512, 1024, 2048, 3072, 4096, 5120]
mat_sizes = [6144, 7128]
n_exec = 100
dtype = torch.float32

cores = os.getenv("OMP_NUM_THREADS")

file_path = dir_test + test_name + ".csv"
# Check if the file exists
if not os.path.exists(file_path):
    # Create the file and write the header line
    with open(file_path, "w") as file:
        file.write("cores, line\n")

with open(file_path, "a") as f:
    writer = csv.writer(f)
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
        writer.writerow([cores, shape, tot_time])