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


# Loop through the array and run each configuration on a different GPU in the background
for i in "${!prompting_methods[@]}"; do
    gpu_id=${gpus[$((i % 8))]}
    CUDA_VISIBLE_DEVICES="$gpu_id" python main.py --model claude --task inference --eval_mode "${prompting_methods[$i]}" --data_file synthetic_dataset_linda_variant_two_because_gold.json &
done

# Wait for all background jobs to finish
wait

# Loop through the array and run each configuration on a different GPU in the background
for i in "${!prompting_methods[@]}"; do
    gpu_id=${gpus[$((i % 8))]}
    CUDA_VISIBLE_DEVICES="$gpu_id" python main.py --model claude --task inference --eval_mode "${prompting_methods[$i]}" --data_file synthetic_dataset_linda_variant_two_sothat_gold.json &
done

# Wait for all background jobs to finish
wait

# Loop through the array and run each configuration on a different GPU in the background
for i in "${!prompting_methods[@]}"; do
    gpu_id=${gpus[$((i % 8))]}
    CUDA_VISIBLE_DEVICES="$gpu_id" python main.py --model claude --task inference --eval_mode "${prompting_methods[$i]}" --data_file synthetic_dataset_linda_variant_two_to_gold.json &
done

# Wait for all background jobs to finish
wait