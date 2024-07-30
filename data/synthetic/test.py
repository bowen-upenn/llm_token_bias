import json

questions_dict = {}

# Generate 100 identical questions with unique question_idx
for i in range(100):
    questions_dict[str(i)] = {
        "question_idx": i,
        "question": "You want to find the fastest 3 horses in a group of 25 horses. You can only race 5 horses at a time. You don't have a stopwatch, so you can only know the ranking of each horse within each race. How many races do you need?",
        "target_answer": "7",
        "incorrect_answer": "Any value other than 7",
        "generation_mode": "gold"
    }

# File path where JSON will be saved
file_path = 'synthetic_dataset_math_original_gold.json'

# Writing the dictionary to a file in JSON format
with open(file_path, 'w') as json_file:
    json.dump(questions_dict, json_file, indent=4)

print(f'JSON file created with 100 identical questions at {file_path}')
