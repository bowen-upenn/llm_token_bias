#!/bin/bash

# Array of prompting methods
prompting_methods=(
    "baseline"
    "zs_cot"
    "os"
    "os_cot"
    "os_bob"
    "os_bob_cot"
    "os_incorrect"
    "os_incorrect_cot"
    "fs"
    "fs_cot"
    "weak_control_zs_cot"
    "weak_control_os_cot"
    "control_zs_cot"
    "control_os_cot"
)

models=(
    'gpt-4-turbo', 
    'gpt-3.5-turbo', 
    'meta-llama-3-8b-instruct', 
    'meta-llama-3-70b-instruct', 
    'claude-3-opus-20240229'
)

gpus=(0 1 2 3)

# Loop through the array and run each configuration on a different GPU in the background
for i in "${!prompting_methods[@]}"; do
    for j in "${!models[@]}"; do
        gpu_id=${gpus[$((i % 4))]}  # This will cycle through 0, 1, 2, 3 for the GPUs
        CUDA_VISIBLE_DEVICES="$gpu_id" python main.py --model "${models[$j]}" --task inference --eval_mode "${prompting_methods[$i]}" --data_file synthetic_dataset_linda_variant_three_baseline.json &
    done
done

# Wait for all background jobs to finish
wait
