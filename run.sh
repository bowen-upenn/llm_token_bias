#!/bin/bash

# Array of prompting methods
variants=(
    'linda_original'
    'linda_variant_one_because'
    'linda_variant_one_sothat'
    'linda_variant_one_to'
    'linda_variant_two_because'
    'linda_variant_three'
    'linda_variant_four'
    'sets_original'
    'sets_original_framing'
)
#    'linda_variant_two_sothat'
#    'linda_variant_two_to'

llms=(
  'gpt-3.5-turbo'
  'gpt-4-turbo'
  'gpt-4o'
  'gemini-1.0-pro-002'
  'gemini-1.5-pro-preview-0409'
  'llama3-70b'
  'llama3-8b'
  'llama-2-70b-chat'
  'claude-3-opus-20240229'
  'claude-3-sonnet-20240229'
  'mistral-large-latest'
  'open-mixtral-8x22b'
)

prompt=(
    'baseline'
    'zs_cot'
    'os'
    'os_cot'
    'os_bob'
    'os_bob_cot'
    'os_incorrect'
    'os_incorrect_cot'
    'fs'
    'fs_cot'
    'fs_no_linda'
    'fs_no_linda_cot'
    'weak_control_zs_cot'
    'weak_control_os_cot'
    'control_zs_cot'
    'control_os_cot'
)


group=(
    'gold'
    'random'
)

gpus=(3 4 5 6 7)


# Loop through the array and run each configuration on a different GPU in the background
for i in "${!variants[@]}"; do
    for j in "${!prompt[@]}"; do
        for k in "${!group[@]}"; do
            gpu_id=${gpus[$((i % 5))]}

            fallacy="linda"  # Default fallacy setting
            # Check if it's one of the last two variants
            if [[ $i -ge $((${#variants[@]} - 2)) ]]; then
                fallacy="sets"
            fi

            CUDA_VISIBLE_DEVICES="$gpu_id" python3 main.py --model gpt-3.5-turbo --fallacy "$fallacy" --task inference --eval_mode "${prompt[$j]}" --data_file synthetic_dataset_"${variants[$i]}"_"${group[$k]}".json &
        done
    done
done

# Wait for all background jobs to finish
wait
