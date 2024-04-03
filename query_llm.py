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
    def __init__(self, args, openai_key=None, all_occupations=None):
        self.args = args
        if openai_key is not None:
            self.api_key = openai_key
        else:
            with open("openai_key.txt", "r") as api_key_file:
                self.api_key = api_key_file.read()
        self.all_occupations = all_occupations


    def query_llm(self, question=None, llm_model='gpt-3.5-turbo', step='answer_question', target_answer=None, model_answer=None, critic=None, grader_id=0, verbose=False):
        client = OpenAI(api_key=self.api_key)

        round = 3 if step == 'generate_data' and self.args['datasets']['generate_mode'] != 'baseline' else 1
        for round_idx in range(round):
            if step == 'answer_question':
                messages = prompt_to_answer_the_question(question)
            elif step == 'grade_answer':
                messages = prompt_to_grade_the_answer(question, target_answer, model_answer, grader_id)
            elif step == 'critic_answer':
                messages = prompt_to_critic_the_answer(question, model_answer)
            elif step == 'reanswer_question':
                messages = prompt_to_reanswer_the_question(question, model_answer, critic)

            elif step == 'generate_data':
                if self.args['datasets']['generate_mode'] == 'baseline':
                    messages = prompt_to_create_linda_problems_baseline()
                elif self.args['datasets']['generate_mode'] == 'gold':
                    if round_idx == 0:
                        messages = prompt_to_write_a_bio()
                    elif round_idx == 1:
                        messages = prompt_to_find_a_hobby(previous_response_bio)
                    else:
                        random_occupation = random.sample(self.all_occupations, 1)[0]
                        messages = prompt_to_create_linda_problems(previous_response_bio, previous_response_hobby, random_occupation)
                elif self.args['datasets']['generate_mode'] == 'random':
                    if round_idx == 0:
                        previous_response_bio = prompt_to_write_a_bio()
                        messages = previous_response_bio
                    elif round_idx == 1:
                        previous_response_hobby = prompt_to_find_a_irrelevant_hobby()
                        messages = previous_response_hobby
                    else:
                        random_occupation = random.sample(self.all_occupations, 1)[0]
                        messages = prompt_to_create_linda_problems_irrelevant(previous_response_bio, previous_response_hobby, random_occupation)
                else:
                    raise ValueError(f'Invalid generate_mode: {self.args["datasets"]["generate_mode"]}')
            else:
                raise ValueError(f'Invalid step: {step}')

            try:
                # print('messages', messages)
                response = client.chat.completions.create(
                    model=llm_model,  # 'gpt-3.5-turbo' or 'gpt-4-turbo-preview'
                    messages=messages,
                    max_tokens=500
                )
                response = response.choices[0].message.content

                if step == 'generate_data' and round_idx == 0:
                    previous_response_bio = response
                elif step == 'generate_data' and round_idx == 1:
                    previous_response_hobby = response
            except:
                response = "Invalid response. "
            if verbose:
                print(f'LLM Response: {response}')

        return response
