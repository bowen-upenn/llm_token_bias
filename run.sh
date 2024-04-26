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

# GPU IDs (assuming 4 GPUs are available, numbered 0-3)
gpus=(0 1 2 3 4 5 6)

# Loop through the array and run each configuration on a different GPU in the background
for i in "${!prompting_methods[@]}"; do
    gpu_id=${gpus[$((i % 7))]}  # This will cycle through 0, 1, 2, 3 for the GPUs
    CUDA_VISIBLE_DEVICES="$gpu_id" python main.py --model gpt4 --task inference --eval_mode "${prompting_methods[$i]}" --data_file synthetic_dataset_linda_variant_four_random.json --verbose &
done

# Wait for all background jobs to finish
wait
