# Hi, I do a matmul using torch

import time
import json
import torch
import ttnn
import numpy as np

MAT_SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]#, 2048]#, 4096]
B = 100
COMPILE_FIRST = True
JUMP_FIRST_RUN = False

torch.manual_seed(42)

device = ttnn.open_device(device_id=0)

grid = device.compute_with_storage_grid_size()  

confs = [{
        "name":                     "hshard_HiFi_f32",
        "memory_config":            ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        "core_grid":                None,
        "enable_program_cache":     False,
        "compute_kernel_config":    None,
        # "max_mat_size":             1024,
        "ttnn_dtype":               ttnn.float32,
        "torch_dtype":              torch.float32,
    },{
        "name":                     "wshard_HiFi_f32",
        "memory_config":            ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        "core_grid":                None,
        "enable_program_cache":     False,
        "compute_kernel_config":    None,
        # "max_mat_size":             1024,
        "ttnn_dtype":               ttnn.float32,
        "torch_dtype":              torch.float32,
    },{
        "name":                     "bshard_HiFi_f32",
        "memory_config":            ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        "core_grid":                None,
        "enable_program_cache":     False,
        "compute_kernel_config":    ttnn.GrayskullComputeKernelConfig(
                                        math_fidelity=ttnn.MathFidelity.HiFi4,
                                        math_approx_mode=False,
                                    ),
        # "max_mat_size":             1024,
        "ttnn_dtype":               ttnn.float32,
        "torch_dtype":              torch.float32,
    },{
        "name":                     "bshard_LoFi_f32",
        "memory_config":            ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        "core_grid":                None,
        "enable_program_cache":     False,
        "compute_kernel_config":    ttnn.GrayskullComputeKernelConfig(
                                        math_fidelity=ttnn.MathFidelity.LoFi,
                                        math_approx_mode=True,
                                    ),
        # "max_mat_size":             1024,
        "ttnn_dtype":               ttnn.float32,
        "torch_dtype":              torch.float32,
    },{
        "name":                     "progcache_f32",
        "memory_config":            ttnn.L1_MEMORY_CONFIG, # faster than ttnn.DRAM_MEMORY_CONFIG
        "core_grid":                None,
        "enable_program_cache":     True,
        "compute_kernel_config":    None,
        # "max_mat_size":             1024,
        "ttnn_dtype":               ttnn.float32,
        "torch_dtype":              torch.float32
    },{
        "name":                     "progcache_f32",
        "memory_config":            ttnn.L1_MEMORY_CONFIG, # faster than ttnn.DRAM_MEMORY_CONFIG
        "core_grid":                ttnn.CoreGrid(y=grid.y, x=grid.x),
        "enable_program_cache":     True,
        "compute_kernel_config":    None,
        # "max_mat_size":             1024,
        "ttnn_dtype":               ttnn.float32,
        "torch_dtype":              torch.float32
    },{
        # Torch is waaaaay slower using bfloat16 (?)
        "name":                     "progcache_bf16",
        "memory_config":            ttnn.L1_MEMORY_CONFIG,
        "core_grid":                ttnn.CoreGrid(y=grid.y, x=grid.x),
        "enable_program_cache":     True,
        "compute_kernel_config":    None,
        # "max_mat_size":             512, # no real limit, just really slow
        "ttnn_dtype":               ttnn.bfloat16,
        "torch_dtype":              torch.bfloat16
    }
]


def mm_torch(a,b):
    start = time.time()
    c = torch.matmul(a, b)
    tot = time.time() - start
    return tot, c

def mm_ttnn(a,b, conf):
    a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn_dtype, device=device)
    b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn_dtype, device=device)
    
    start = time.time()
    c = ttnn.matmul(
        a,
        b,
        memory_config=          conf['memory_config'],
        compute_kernel_config=  conf['compute_kernel_config'],
        core_grid=              conf['core_grid'],
    )
    tot = time.time() - start 

    return tot, c

def check_close(torch_tensor, ttnn_tensor, atol=0.1):
    ttnn_tensor = torch.Tensor(ttnn.to_torch(ttnn_tensor))
    equals = torch.sum(torch.isclose(torch_tensor, ttnn_tensor, atol=atol))
    perc = equals/torch_tensor.numel() * 100
    print(f"Close values: {perc:.3f}% ({torch_tensor.numel()})")


# mat_sizes = [mt for mt in MAT_SIZES if mt <= CONF['max_mat_size']]
mat_sizes = MAT_SIZES

results = {
    "mat_sizes": mat_sizes,
}

# Gather sample results in torch
l_time_torch = list()
for mat_size in mat_sizes:
    l_avg_torch = list()
    N, K, M = mat_size, mat_size, mat_size
    
    for i in range(B):
        a = torch.randn((N, K), dtype=torch.float32)
        b = torch.randn((K, M), dtype=torch.float32)

        time_torch, c_torch = mm_torch(a, b)

        if JUMP_FIRST_RUN and i == 0:
            continue
        l_avg_torch.append(time_torch)

    l_time_torch.append(np.mean(l_avg_torch))


results["torch_f32"] = l_time_torch


for conf in confs:
    ttnn.close_device(device)
    device = ttnn.open_device(device_id=0)
    if conf['enable_program_cache']:
        ttnn.enable_program_cache(device)
    print(f"Running config {conf['name']}")
    torch_dtype=conf['torch_dtype']
    ttnn_dtype=conf['ttnn_dtype']

    if COMPILE_FIRST:
        a = torch.randn((2, 2), dtype=torch_dtype)
        b = torch.randn((2, 2), dtype=torch_dtype)
        mm_ttnn(a, b, conf)
    
    l_time_ttnn = list()
    exploded = False
    for mat_size in mat_sizes:
        if exploded: # or mat_size > conf['max_mat_size']:
                l_time_ttnn.append(None)
                continue
        try:
            l_avg_ttnn = list()
            N, K, M = mat_size, mat_size, mat_size
            
            print(f"[{N},{K}] @ [{K},{N}]")
            for i in range(B):
                a = torch.randn((N, K), dtype=torch_dtype)
                b = torch.randn((K, M), dtype=torch_dtype)

                time_ttnn, c_ttnn = mm_ttnn(a, b, conf)

                if JUMP_FIRST_RUN and i == 0:
                    continue

                l_avg_ttnn.append(time_ttnn)

            l_time_ttnn.append(np.mean(l_avg_ttnn))
        except:
            print("Exploded!")
            exploded = True
            l_time_ttnn.append(None)
            continue

    results[conf['name']] = l_time_ttnn

with open("results_compare.json", "w") as f:
    json.dump(
        results, f
    )
    

ttnn.close_device(device)