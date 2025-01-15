# Run with tt-metal env

import time
import json
import torch
import ttnn
import numpy as np

MAT_SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
B = 100

torch.manual_seed(42)

device = ttnn.open_device(device_id=0)

grid = device.compute_with_storage_grid_size()  


c_bs = {
    "memory_config":            ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    "core_grid":                None,
    "enable_program_cache":     False,
    "compute_kernel_config":    ttnn.GrayskullComputeKernelConfig(
                                    math_fidelity=ttnn.MathFidelity.LoFi, # Changes just a bit
                                    math_approx_mode=True,
                                ),
    "max_mat_size":             1024,
    "ttnn_dtype":               ttnn.float32,
    "torch_dtype":              torch.float32,
}

cbs = {
    "memory_config":            ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    "core_grid":                ttnn.CoreGrid(y=grid.y, x=grid.x),
    "enable_program_cache":     False,
    "compute_kernel_config":    ttnn.GrayskullComputeKernelConfig(
                                    math_fidelity=ttnn.MathFidelity.LoFi, # Changes just a bit
                                    math_approx_mode=True,
                                ),
    "max_mat_size":             1024,
    "ttnn_dtype":               ttnn.float32,
    "torch_dtype":              torch.float32,
}


c2 = {
    "memory_config":            ttnn.L1_MEMORY_CONFIG, # faster than ttnn.DRAM_MEMORY_CONFIG
    "core_grid":                ttnn.CoreGrid(y=grid.y, x=grid.x),
    "enable_program_cache":     True,
    "compute_kernel_config":    None,
    "max_mat_size":             1024,
    "ttnn_dtype":               ttnn.float32,
    "torch_dtype":              torch.float32
}

c3 = {
    # Torch is waaaaay slower using bfloat16 (?)
    "memory_config":            ttnn.L1_MEMORY_CONFIG,
    "core_grid":                ttnn.CoreGrid(y=grid.y, x=grid.x),
    "enable_program_cache":     True,
    "compute_kernel_config":    None,
    "max_mat_size":             512, # no real limit, just really slow
    "ttnn_dtype":               ttnn.bfloat16,
    "torch_dtype":              torch.bfloat16
}


CONF = cbs


ttnn_dtype  =   CONF['ttnn_dtype']
torch_dtype =   CONF['torch_dtype']


if CONF['enable_program_cache']:
    ttnn.enable_program_cache(device)


def mm_torch(a,b):
    start = time.time()
    c = torch.matmul(a, b)
    tot = time.time() - start
    return tot, c

def mm_ttnn(a,b):
    a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn_dtype, device=device)
    b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn_dtype, device=device)
    
    start = time.time()
    c = ttnn.matmul(
        a,
        b,
        memory_config=CONF['memory_config'],
        compute_kernel_config=CONF['compute_kernel_config'],
        core_grid=CONF['core_grid'],
    )
    tot = time.time() - start 

    return tot, c

def check_close(torch_tensor, ttnn_tensor, atol=0.1):
    ttnn_tensor = torch.Tensor(ttnn.to_torch(ttnn_tensor))
    equals = torch.sum(torch.isclose(torch_tensor, ttnn_tensor, atol=atol))
    perc = equals/torch_tensor.numel() * 100
    print(f"Close values: {perc:.3f}% ({torch_tensor.numel()})")


mat_sizes = [mt for mt in MAT_SIZES if mt <= CONF['max_mat_size']]
 

l_time_torch = list()
l_time_ttnn = list()
for mat_size in mat_sizes:
    l_avg_torch = list()
    l_avg_ttnn = list()
    N, K, M = mat_size, mat_size, mat_size
    print(f"[{N},{K}] @ [{K},{N}]")
    for i in range(B):
        a = torch.randn((N, K), dtype=torch_dtype)
        b = torch.randn((K, M), dtype=torch_dtype)
        
        time_torch, c_torch = mm_torch(a,b)
        time_ttnn, c_ttnn = mm_ttnn(a,b)
        # check_close(c_torch, c_ttnn)
        if i > -1:
            l_avg_torch.append(time_torch)
            l_avg_ttnn.append(time_ttnn)

    l_time_torch.append(np.mean(l_avg_torch))
    l_time_ttnn.append(np.mean(l_avg_ttnn))


with open("results.json", "w") as f:
    json.dump({
            "mat_sizes": mat_sizes,
            "torch_time": l_time_torch,
            "ttnn_time": l_time_ttnn,
        },
        f
    )
    

ttnn.close_device(device)

# print(f"Time torch: {tot_time_torch :.10f}")
# print(f"Time ttnn: {tot_time_ttnn :.10f}")

