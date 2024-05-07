#!/bin/bash

# Array of prompting methods
variants=(
    'one_because'
    'one_sothat'
    'one_to'
    'two_because'
    'two_sothat'
    'two_to'
    'three'
    'four'
)


gpus=(0 1 2 3 4 7)


# Loop through the array and run each configuration on a different GPU in the background
for i in "${!variants[@]}"; do    gpu_id=${gpus[$((i % 6))]}
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 main.py --model gpt-4-turbo --task inference --eval_mode zs_cot --data_file synthetic_dataset_linda_variant_"${variants[$i]}"_random.json &
done

# Wait for all background jobs to finish
wait


##!/bin/bash
#
## Array of prompting methods
#prompting_methods=(
#    "baseline"
#    "zs_cot"
#    "os"
#    "os_cot"
#    "os_bob"
#    "os_bob_cot"
#    "os_incorrect"
#    "os_incorrect_cot"
#    "fs"
#    "fs_cot"
#    "weak_control_zs_cot"
#    "weak_control_os_cot"
#    "control_zs_cot"
#    "control_os_cot"
#)
#
#
#gpus=(0 1 2 3 4 7)
#
#
## Loop through the array and run each configuration on a different GPU in the background
#for i in "${!prompting_methods[@]}"; do    gpu_id=${gpus[$((i % 6))]}
#    CUDA_VISIBLE_DEVICES="$gpu_id" python3 main.py --model gpt-4-turbo --task inference --eval_mode "${prompting_methods[$i]}" --data_file synthetic_dataset_linda_original_gold.json &
#done
#
## Wait for all background jobs to finish
#wait