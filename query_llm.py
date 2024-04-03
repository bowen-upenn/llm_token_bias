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


    def query_llm(self, question, llm_model='gpt-3.5-turbo', step='answer_question', target_answer=None, model_answer=None, critic=None, grader_id=0, verbose=False):
        client = OpenAI(api_key=self.api_key)

        if step == 'answer_question':
            messages = prompt_to_answer_the_question(question)
        elif step == 'grade_answer':
            messages = prompt_to_grade_the_answer(question, target_answer, model_answer, grader_id)
        elif step == 'critic_answer':
            messages = prompt_to_critic_the_answer(question, model_answer)
        elif step == 'reanswer_question':
            messages = prompt_to_reanswer_the_question(question, model_answer, critic)
        elif step == 'generate_data':
            messages = prompt_to_generate_synthetic_data(question)
        else:
            raise ValueError(f'Invalid step: {step}')

        try:
            response = client.chat.completions.create(
                model=llm_model,  # 'gpt-3.5-turbo' or 'gpt-4-turbo-preview'
                messages=messages,
            )
            response = response.choices[0].message.content
        except:
            response = "Invalid response. "
        if verbose:
            print(f'LLM Response at step {step}: {response}')

        return response
