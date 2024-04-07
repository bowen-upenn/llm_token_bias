import torch
import numpy as np
from tqdm import tqdm
import os
import json
import openai
from openai import OpenAI
import random

from utils import *
from prompts import *


class QueryLLM:
    def __init__(self, args, openai_key=None):
        self.args = args
        if openai_key is not None:
            self.api_key = openai_key
        else:
            with open("openai_key.txt", "r") as api_key_file:
                self.api_key = api_key_file.read()
        self.AllPrompts = AllPrompts(args)


    def query_llm(self, question=None, llm_model='gpt-3.5-turbo', step='answer_question', target_answer=None, model_answer=None, critic=None, grader_id=0, verbose=False):
        client = OpenAI(api_key=self.api_key)
        linda_problem_variant = self.args['datasets']['linda_problem_variant']
        connector = self.args['datasets']['connector']

        if step == 'generate_data' and self.args['datasets']['generate_mode'] != 'baseline':
            if linda_problem_variant == 'original':
                self.AllPrompts.select_a_random_occupation()
                self.AllPrompts.select_a_random_gender()
                self.AllPrompts.select_a_random_age()
                self.AllPrompts.select_a_random_race()
            elif linda_problem_variant == 'variant_one':
                self.AllPrompts.select_a_random_roc_story()
                self.AllPrompts.connector = connector
            elif linda_problem_variant == 'variant_two':
                self.AllPrompts.select_a_random_news()
                self.AllPrompts.connector = connector
            elif linda_problem_variant == 'variant_three':
                self.AllPrompts.select_a_random_gender()
                self.AllPrompts.select_a_random_age()
                self.AllPrompts.select_a_random_race()
                self.AllPrompts.select_a_random_disease_symptom_pair()

        if step == 'generate_data' and self.args['datasets']['generate_mode'] != 'baseline':
            if linda_problem_variant == 'original':
                round = 5
            elif linda_problem_variant == 'variant_one':
                round = 3
            elif linda_problem_variant == 'variant_two':
                round = 2
            elif linda_problem_variant == 'variant_three':
                round = 2
            else:
                round = 1
        else:
            round = 1

        for round_idx in range(round):
            if step == 'answer_question':
                messages = self.AllPrompts.prompt_to_answer_the_question(question)
            elif step == 'grade_answer':
                messages = self.AllPrompts.prompt_to_grade_the_answer(question, target_answer, model_answer, grader_id)
            elif step == 'critic_answer':
                messages = self.AllPrompts.prompt_to_critic_the_answer(question, model_answer)
            elif step == 'reanswer_question':
                messages = self.AllPrompts.prompt_to_reanswer_the_question(question, model_answer, critic)

            elif step == 'generate_data':
                if self.args['datasets']['generate_mode'] == 'baseline':
                    messages = self.AllPrompts.prompt_to_create_linda_problems_baseline()
                else:
                    # the following codes carefully curate the synthetic data generation process for different variations of the Linda problems
                    if linda_problem_variant == 'variant_one':
                        if round_idx == 0:
                            messages = self.AllPrompts.prompt_to_extend_the_story()
                        elif round_idx == 1:  # generate golden examples
                            messages = self.AllPrompts.prompt_to_create_linda_problems_variant_one(previous_response_extension)
                        else:   # generate random examples
                            messages = self.AllPrompts.prompt_to_create_linda_problems_variant_one_irrelevant(previous_response_extension, previous_response_completion)  # same previous_response_extension

                    elif linda_problem_variant == 'variant_two':
                        if round_idx == 0:
                            messages = self.AllPrompts.prompt_to_create_linda_problems_variant_two()
                        else:
                            messages = self.AllPrompts.prompt_to_create_linda_problems_variant_two_irrelevant(previous_response_completion)

                    elif linda_problem_variant == 'variant_three':
                        if round_idx == 0:
                            messages = self.AllPrompts.prompt_to_create_linda_problems_variant_three()
                        else:
                            messages = self.AllPrompts.prompt_to_create_linda_problems_variant_three_irrelevant()

                    else: # default linda_problem_variant == 'original':
                        if round_idx == 0:
                            messages = self.AllPrompts.prompt_to_write_a_bio()
                        elif round_idx == 1:
                            messages = self.AllPrompts.prompt_to_find_a_hobby(previous_response_bio)
                        elif round_idx == 2:
                            messages = self.AllPrompts.prompt_to_find_a_irrelevant_hobby()
                        elif round_idx == 3:
                            messages = self.AllPrompts.prompt_to_create_linda_problems_original(previous_response_bio, previous_response_hobby)
                        else:
                            messages = self.AllPrompts.prompt_to_create_linda_problems_original_irrelevant(previous_response_bio, previous_response_hobby_irrelevant)
            else:
                raise ValueError(f'Invalid step: {step}')

            # try:
            # print('messages', messages)
            response = client.chat.completions.create(
                model=llm_model,  # 'gpt-3.5-turbo' or 'gpt-4-turbo-preview'
                messages=messages,
                max_tokens=500
            )
            response = response.choices[0].message.content

            # record variables useful for upcoming rounds
            if step == 'generate_data':
                if linda_problem_variant == 'original':
                    if round_idx == 0:
                        previous_response_bio = response
                    elif round_idx == 1:
                        previous_response_hobby = response
                    elif round_idx == 2:
                        previous_response_hobby_irrelevant = response
                    elif round_idx == 3:
                        linda_problem_gold = response
                    else:
                        linda_problem_random = response

                elif linda_problem_variant == 'variant_one':
                    if round_idx == 0:
                        previous_response_extension = response
                    elif round_idx == 1:
                        previous_response_completion = response
                        response = ' ' + response if response[0] != ' ' else response
                        linda_problem_gold = self.AllPrompts.random_roc_story + "\nWhich is more likely?\n(a) " + previous_response_extension \
                                             + "\n(b) " + previous_response_extension[:-1] + " " + connector + response
                    else:
                        response = ' ' + response if response[0] != ' ' else response
                        linda_problem_random = self.AllPrompts.random_roc_story + "\nWhich is more likely?\n(a) " + previous_response_extension \
                                               + "\n(b) " + previous_response_extension[:-1] + " " + connector + response

                elif linda_problem_variant == 'variant_two':
                    if round_idx == 0:
                        previous_response_completion = response
                        response = ' ' + response if response[0] != ' ' else response
                        linda_problem_gold = self.AllPrompts.random_news_before_last_sentence + "\nWhich is more likely?\n(a) " + self.AllPrompts.random_news_last_sentence \
                                             + "\n(b) " + self.AllPrompts.random_news_last_sentence[:-1] + " " + connector + response
                    else:
                        response = ' ' + response if response[0] != ' ' else response
                        linda_problem_random = self.AllPrompts.random_news_before_last_sentence + "\nWhich is more likely?\n(a) " + self.AllPrompts.random_news_last_sentence \
                                               + "\n(b) " + self.AllPrompts.random_news_last_sentence[:-1] + " " + connector + response

                elif linda_problem_variant == 'variant_three':
                    if round_idx == 0:
                        linda_problem_gold = response
                    else:
                        linda_problem_random = response

            # except:
            #     response = "Invalid response. "
            if verbose:
                print(f'LLM Response: {response}')

        return linda_problem_gold, linda_problem_random
