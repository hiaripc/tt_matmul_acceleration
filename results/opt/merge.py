import pandas as pd 

df = pd.read_csv("./results/matmul2d_1.0.csv")
df_info_fr = pd.read_csv("./results/matmul2d_first_run_1.0.csv")

df = df.copy()
for col in df_info_fr.columns:
    df[col] = df_info_fr[col]

df.drop(["k", "n", "grid_size"], inplace=True, axis=1)

df = df[['conf', 'm', 'use_trace', 'in0_sharded', 'out_sharded',
       'in0_storage_type', 'in1_storage_type', 'out_storage_type', 'dtype',
       'math_fidelity', 'transfer_time_in0', 'transfer_time_in1', 
       'kernel_config_time', 'first_run_time', 'second_run_time',
       'compile_time', 'inference_time_avg', 'trace_time','TFLOPs (avg)',
       'Utilization (vs user grid)', 'Utilization (vs 8x8 full grid)']]

# df.to_csv(..)