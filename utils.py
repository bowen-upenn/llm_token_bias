import torch
import numpy as np
import json
import re
import os
import csv
import pandas as pd


class Grader:
    def __init__(self):
        self.count_correct = 0
        self.count_incorrect = 0
        self.count_total = 0

    def average_score(self):
        """Calculate and return the average score of the grades."""
        if self.count_total == 0:
            return 0, 0, None  # Return 0 if there are no grades to avoid division by zero

        accuracy = self.count_correct / self.count_total

        stat = {
            'count_correct': self.count_correct,
            'count_incorrect': self.count_incorrect,
            'count_total': self.count_total
        }
        return accuracy, stat

    def accumulate_grades(self, args, grades):
        # accumulate the grades
        count_match_correct = 0
        for grade in grades:
            grade = grade.lower()
            if re.search(r'\[correct]', grade) or (re.search("correct", grade) and not re.search("incorrect", grade)):
                count_match_correct += 1

        match_correct = True if count_match_correct >= 2 else False  # majority vote: if at least 2 out of 3 graders agree, the answer is correct

        if match_correct:
            majority_vote = 'Majority vote is [Correct] with a score of ' + str(count_match_correct)
            if args['inference']['verbose']:
                print(f'{Colors.OKBLUE}{majority_vote}{Colors.ENDC}')
        else:
            majority_vote = 'Majority vote is [Incorrect] with a score of ' + str(count_match_correct)
            if args['inference']['verbose']:
                print(f'{Colors.FAIL}{majority_vote}{Colors.ENDC}')

        self.count_total += 1
        if match_correct:
            self.count_correct += 1
        else:
            self.count_incorrect += 1

        return majority_vote


def print_response(retry, grader, batch_count, len_test_loader, output_response_filename):
    init_answer_accuracy, init_stats = grader[0].average_score()
    if retry:
        retry_answer_accuracy, retry_stats = grader[1].average_score()
        if (batch_count + 1) == len(test_loader):
            print('Accuracy at batch idx ', batch_count, ': init', init_answer_accuracy, init_stats, 'retry', retry_answer_accuracy, retry_stats)
            record_final_accuracy(output_response_filename, init_answer_accuracy, init_stats, retry_answer_accuracy, retry_stats)
        else:
            print('Accuracy at batch idx ', batch_count, ': init', init_answer_accuracy, 'retry', retry_answer_accuracy)
    else:
        if (batch_count + 1) == len_test_loader:
            print('Accuracy at batch idx ', batch_count, ':', init_answer_accuracy, init_stats)
            record_final_accuracy(output_response_filename, init_answer_accuracy, init_stats)
        else:
            print('Accuracy at batch idx ', batch_count, ':', init_answer_accuracy)


def write_response_to_json(question_id, response_dict, output_response_filename, fallacy_type=None, generation_mode=None, logical_connector=None, linda_problem_variant=None):
    # Check if the JSON file already exists
    if fallacy_type is not None:
        output_response_filename = output_response_filename + '_' + fallacy_type + '_' + linda_problem_variant
    if linda_problem_variant == 'variant_one' or linda_problem_variant == 'variant_two':
        output_response_filename = output_response_filename + '_' + logical_connector.replace(" ", "")
    output_response_filename = output_response_filename + '_' + generation_mode + '.json'

    if os.path.exists(output_response_filename):
        # Read the existing content
        with open(output_response_filename, 'r') as file:
            data = json.load(file)
    else:
        # Initialize an empty list if the file doesn't exist
        data = {}

    # Append the new response
    if isinstance(question_id, int):
        data[str(question_id)] = response_dict
    elif isinstance(question_id, str):
        data[question_id] = response_dict
    else:
        data[str(question_id.item())] = response_dict

    # Write the updated data back to the file
    with open(output_response_filename, 'w') as file:
        json.dump(data, file, indent=4)


def record_final_accuracy(output_response_filename, final_accuracy, stats, final_retry_accuracy=None, retry_stats=None):
    # Assuming the JSON file exists at this point
    with open(output_response_filename, 'r') as file:
        data = json.load(file)

    # Add the accuracy to the JSON data
    data['final_accuracy'] = str(final_accuracy)
    data['stats'] = stats
    data['final_retry_accuracy'] = str(final_retry_accuracy)
    data['retry_stats'] = retry_stats

    # Write the updated data back to the file
    with open(output_response_filename, 'w') as file:
        json.dump(data, file, indent=4)


class Colors:
    HEADER = '\033[95m'  # Purple
    OKBLUE = '\033[94m'  # Blue
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'    # Red
    ENDC = '\033[0m'     # Reset color


def load_occupations(filename):
    # data source: https://www.bls.gov/oes/current/oes_stru.htm#39-0000
    all_occupations = []
    with open(filename, 'r') as file:
        for line in file:
            # Split the line into parts based on whitespace and then rejoin from the second element to get the occupation name
            occupation_name = ' '.join(line.split()[1:])
            occupation_name = occupation_name.split(',', 1)[0].strip().lower()
            all_occupations.append(occupation_name)

    return all_occupations


def load_roc_stories(filename):
    # data source: https://cs.rochester.edu/nlp/rocstories/
    stories = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Concatenating the sentences for each story
            story = f"{row['sentence1']} {row['sentence2']} {row['sentence3']} {row['sentence4']} {row['sentence5']}"
            stories.append(story)
    return stories


def load_cnn_dailymails(filename):
    # data source: https://huggingface.co/datasets/cnn_dailymail
    df = pd.read_parquet(filename)
    news = df['highlights'].tolist()
    return news


def load_disease_symptoms(filename):
    # data source: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset?select=symptom_Description.csv
    df = pd.read_csv(filename)
    return df