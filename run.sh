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


gpus=(0 1 2 3 4 5 6 7)

models=(
    "meta-llama-3-70b-instruct"
    "gpt-4-turbo"
)

# Loop through the array and run each configuration on a different GPU in the background
for i in "${!prompting_methods[@]}"; do
    gpu_id=${gpus[$((i % 8))]}
    for j in "${!models[@]}"; do
        CUDA_VISIBLE_DEVICES="$gpu_id" python main.py --model "${models[$j]}" --task inference --eval_mode "${prompting_methods[$i]}" --data_file synthetic_dataset_linda_variant_three_gold.json &
    done
done

# Wait for all background jobs to finish
wait