import torch
import numpy as np
from tqdm import tqdm
import os
import json
import openai
from openai import OpenAI
import random
from utils import *


class QueryLLM:
    def __init__(self, args, openai_key=None):
        self.args = args
        if openai_key is not None:
            self.api_key = openai_key
        else:
            with open("openai_key.txt", "r") as api_key_file:
                self.api_key = api_key_file.read()

    def prompt_to_answer_the_question(self, question):
        message = [
            {"role": "system", "content": ""},  # add instruction here
            {"role": "user", "content": ""}     # add question here
        ]
        return message

    def prompt_to_critic_the_answer(self, question, model_answer):
        message = [
            {"role": "system", "content": ""},  # add instruction and question here
            {"role": "user", "content": ""}     # add model response here
        ]   # Note: ask the model to attach special tokens in the response, such as [Yes] or [No]
        return message

    def prompt_to_reanswer_the_question(self, question, init_model_answer, critic):
        message = [
            {"role": "system", "content": ""},  # add initial instruction here
            {"role": "user", "content": ""},    # add initial question here
            {"role": "system", "content": ""},  # add initial model response here
            {"role": "user", "content": ""},    # add critic here
            {"role": "system", "content": ""}   # add reattempt instruction here
        ]
        return message

    def prompt_to_grade_the_answer(self, question, target_answer, model_answer, grader_id=0):
        if grader_id == 0:
            messages = [
                {"role": "system", "content": ""},  # add instruction here
                {"role": "user", "content": ""},    # add model response here
            ]
        elif grader_id == 1:
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": ""},
            ]
        else:
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": ""},
            ]
        return messages

    def prompt_to_generate_synthetic_data(self, question):
        message = [
            {"role": "system", "content": ""},  # add instruction here
            {"role": "user", "content": ""}     # add in-context learning example here
        ]
        return message


    def query_llm(self, question, llm_model='gpt-3.5-turbo', step='answer_question', target_answer=None, model_answer=None, critic=None, grader_id=0, verbose=False):
        client = OpenAI(api_key=self.api_key)

        if step == 'answer_question':
            messages = self.prompt_to_answer_the_question(question)
        elif step == 'grade_answer':
            messages = self.prompt_to_grade_the_answer(question, target_answer, model_answer, grader_id)
        elif step == 'critic_answer':
            messages = self.prompt_to_critic_the_answer(question, model_answer)
        elif step == 'reanswer_question':
            messages = self.prompt_to_reanswer_the_question(question, model_answer, critic)
        elif step == 'generate_data':
            messages = self.prompt_to_generate_synthetic_data(question)
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
