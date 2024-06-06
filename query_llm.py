import torch
import numpy as np
from tqdm import tqdm
import os
import json
import random
import re

# OpenAI ChatGPT API
import openai
from openai import OpenAI

# Google Gemini API from VertexAI
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession

# Meta Llama API from Replicate
import replicate

# Anthropic Claude API
import anthropic

# Mistral API
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from utils import *
from data_prompts import *
from inference_prompts import *


class QueryLLM:
    def __init__(self, args):
        self.args = args

        # Load API keys or tokens
        with open("api_tokens/openai_key.txt", "r") as api_key_file:
            self.api_key = api_key_file.read()
        if re.search(r'gemini', self.args['models']['llm_model']) is not None:
            with open("api_tokens/gemini_project_id.txt", "r") as vertexai_project_id_file:
                self.project_id = vertexai_project_id_file.read()
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.args['models']['gemini_credential_path']
        elif re.search(r'llama', self.args['models']['llm_model']) is not None:
            with open("api_tokens/llama_key.txt", "r") as llama_key_file:
                llama_key = llama_key_file.read()
            os.environ['REPLICATE_API_TOKEN'] = llama_key
        elif re.search(r'claude', self.args['models']['llm_model']) is not None:
            with open("api_tokens/claude_key.txt", "r") as claude_key_file:
                self.claude_key = claude_key_file.read()
        elif re.search(r'mistral', self.args['models']['llm_model']) is not None:
            with open("api_tokens/mistral_key.txt", "r") as mistral_key_file:
                self.mistral_key = mistral_key_file.read()

        self.AllDataPrompts = AllDataPrompts(args)
        self.AllInferencePrompts = AllInferencePrompts(args)

        if self.args['inference']['mode'] == 'fs' or self.args['inference']['mode'] == 'fs_cot':
            self.AllInferencePrompts.load_all_data_entries()   # load it only once


    def query_llm(self, question=None, llm_model='gpt-3.5-turbo', step='answer_question', target_answer=None, model_answer=None, critic=None, grader_id=0, verbose=False):
        linda_problem_variant = self.args['datasets']['linda_problem_variant']
        connector = self.args['datasets']['connector']

        ############################### DATA GENERATION ########################################
        if step == 'generate_data' and self.args['datasets']['generate_mode'] != 'baseline':
            if self.args['datasets']['fallacy_type'] == 'linda':
                if linda_problem_variant == 'original':
                    self.AllDataPrompts.select_a_random_occupation()
                    self.AllDataPrompts.select_a_random_gender()
                    self.AllDataPrompts.select_a_random_age()
                    self.AllDataPrompts.select_a_random_race()
                elif linda_problem_variant == 'variant_one':
                    self.AllDataPrompts.select_a_random_roc_story()
                    self.AllDataPrompts.connector = connector
                elif linda_problem_variant == 'variant_two':
                    self.AllDataPrompts.select_a_random_news()
                    self.AllDataPrompts.connector = connector
                elif linda_problem_variant == 'variant_three':
                    self.AllDataPrompts.select_a_random_gender()
                    self.AllDataPrompts.select_a_random_age()
                    self.AllDataPrompts.select_a_random_race()
                    self.AllDataPrompts.select_a_random_disease_symptom_pair()
                elif linda_problem_variant == 'variant_four':
                    self.AllDataPrompts.select_a_random_celebrity()
                elif linda_problem_variant == 'variant_five':
                    self.AllDataPrompts.select_a_random_natural_disaster()
                    self.AllDataPrompts.select_a_random_year()
                    self.AllDataPrompts.select_a_random_gender()
                    self.AllDataPrompts.select_a_random_age()
                    self.AllDataPrompts.select_a_random_race()
                elif linda_problem_variant == 'variant_six':
                    self.AllDataPrompts.select_random_letters()
            elif self.args['datasets']['fallacy_type'] == 'sets':
                self.AllDataPrompts.select_a_random_object()
                self.AllDataPrompts.select_a_random_news_agency()
                self.AllDataPrompts.select_a_random_university()
            else:
                assert False, "Invalid fallacy type."

        if step == 'generate_data' and self.args['datasets']['generate_mode'] != 'baseline':
            if self.args['datasets']['fallacy_type'] == 'linda':
                if linda_problem_variant == 'original':
                    round = 5
                elif linda_problem_variant == 'variant_one':
                    round = 3
                elif linda_problem_variant == 'variant_two':
                    round = 2
                elif linda_problem_variant == 'variant_three':
                    round = 2
                elif linda_problem_variant == 'variant_four':
                    #round = 6
                    round = 2
                elif linda_problem_variant == 'variant_five':
                    round = 4
                elif linda_problem_variant == 'variant_six':
                    round = 1
                else:
                    round = 1
            elif self.args['datasets']['fallacy_type'] == 'sets':
                round = 3
            else:
                assert False, "Invalid fallacy type."
        else:
            round = 1

        for round_idx in range(round):
            ############################### INFERENCE ########################################
            if step == 'answer_question':
                if self.args['inference']['mode'] == 'zs_cot':
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_zero_shot_cot(question)
                elif self.args['inference']['mode'] == 'os':
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_one_shot(question)
                elif self.args['inference']['mode'] == 'os_cot':
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_one_shot_cot(question)
                elif self.args['inference']['mode'] == 'os_bob':
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_one_shot_bob(question)
                elif self.args['inference']['mode'] == 'os_bob_cot':
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_one_shot_bob_cot(question)
                elif self.args['inference']['mode'] == 'os_incorrect':
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_one_shot_incorrect_answer(question)
                elif self.args['inference']['mode'] == 'os_incorrect_cot':
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_one_shot_incorrect_answer_cot(question)
                elif self.args['inference']['mode'] == 'fs':
                    self.AllInferencePrompts.select_random_few_shot_exemplars(self.args['inference']['num_few_shots_exemplars'])
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_few_shots(question)
                elif self.args['inference']['mode'] == 'fs_cot':
                    self.AllInferencePrompts.select_random_few_shot_exemplars(self.args['inference']['num_few_shots_exemplars'])
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_few_shots_cot(question)
                elif self.args['inference']['mode'] == 'self_reflect':
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_self_reflection(question)
                elif self.args['inference']['mode'] == 'weak_control_zs_cot':
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_weak_control_zero_shot_cot(question)
                elif self.args['inference']['mode'] == 'weak_control_os_cot':
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_weak_control_one_shot_cot(question)
                elif self.args['inference']['mode'] == 'control_zs_cot':
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_control_zero_shot_cot(question)
                elif self.args['inference']['mode'] == 'control_os_cot':
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_control_one_shot_cot(question)
                else:
                    messages = self.AllInferencePrompts.prompt_to_answer_the_question_directly(question)

            elif step == 'grade_answer':
                messages = self.AllInferencePrompts.prompt_to_grade_the_answer(target_answer, model_answer, grader_id)
            elif step == 'critic_answer':
                messages = self.AllInferencePrompts.prompt_to_critic_the_answer(question, model_answer)
            elif step == 'reanswer_question':
                messages = self.AllInferencePrompts.prompt_to_reanswer_the_question(question, model_answer, critic)
            elif step == 'extract_answer':
                messages = self.AllInferencePrompts.prompt_to_extract_the_answer(model_answer)


            ############################### DATA GENERATION ########################################
            elif step == 'generate_data':
                if self.args['datasets']['fallacy_type'] == 'linda':
                    if self.args['datasets']['generate_mode'] == 'baseline':
                        messages = self.AllDataPrompts.prompt_to_create_linda_problems_baseline()
                    else:
                        # the following codes carefully curate the synthetic data generation process for different variations of the Linda problems
                        if linda_problem_variant == 'variant_one':
                            if round_idx == 0:
                                messages = self.AllDataPrompts.prompt_to_extend_the_story()
                            elif round_idx == 1:  # generate golden examples
                                messages = self.AllDataPrompts.prompt_to_create_linda_problems_variant_one(previous_response_extension)
                            else:   # generate random examples
                                messages = self.AllDataPrompts.prompt_to_create_linda_problems_variant_one_irrelevant(previous_response_extension, previous_response_completion)  # same previous_response_extension

                        elif linda_problem_variant == 'variant_two':
                            if round_idx == 0:
                                messages = self.AllDataPrompts.prompt_to_create_linda_problems_variant_two()
                            else:
                                messages = self.AllDataPrompts.prompt_to_create_linda_problems_variant_two_irrelevant(previous_response_completion)

                        elif linda_problem_variant == 'variant_three':
                            if round_idx == 0:
                                messages = self.AllDataPrompts.prompt_to_create_linda_problems_variant_three()
                            else:
                                messages = self.AllDataPrompts.prompt_to_create_linda_problems_variant_three_irrelevant()

                        elif linda_problem_variant == 'variant_four':
                            if round_idx == 0:
                                messages = self.AllDataPrompts.prompt_celebrity_few_shot()
                            else:
                                messages = self.AllDataPrompts.get_random_name_same_gender_as_celebrity(linda_problem_gold)
                                print(f"messages: {messages}")
                            '''
                            if round_idx == 0:
                                messages = self.AllDataPrompts.prompt_to_write_an_event()
                            elif round_idx == 1:
                                messages = self.AllDataPrompts.prompt_to_write_an_achievement(previous_response_event)
                            elif round_idx == 2:
                                messages = self.AllDataPrompts.prompt_to_find_a_small_failure(previous_response_event)
                            elif round_idx == 3:
                                messages = self.AllDataPrompts.prompt_to_create_linda_problems_variant_four(previous_response_event, previous_response_achievement, previous_response_failure)
                            elif round_idx == 4:
                                messages = self.AllDataPrompts.prompt_to_create_linda_problems_variant_four_irrelevant(previous_response_event, previous_response_achievement, previous_response_failure, previous_response_problem)
                            else:
                                messages = self.AllDataPrompts.prompt_to_create_linda_problems_variant_four_nobody(previous_response_event, previous_response_achievement, previous_response_failure, previous_response_problem)
                            '''

                        elif linda_problem_variant == 'variant_five':
                            if round_idx == 0:
                                messages = self.AllDataPrompts.prompt_to_write_a_disaster()
                            elif round_idx == 1:
                                messages = self.AllDataPrompts.prompt_to_write_another_related_disaster(previous_response_disaster)
                            elif round_idx == 2:
                                messages = self.AllDataPrompts.prompt_to_create_linda_problems_variant_five(previous_response_disaster, previous_response_disaster_related)
                            else:
                                messages = self.AllDataPrompts.prompt_to_create_linda_problems_variant_five_irrelevant(previous_response_disaster, previous_response_disaster_related, previous_response_problem)

                        elif linda_problem_variant == 'variant_six':
                            messages = self.AllDataPrompts.prompt_to_create_linda_problems_variant_six()

                        else: # default linda_problem_variant == 'original':
                            if round_idx == 0:
                                messages = self.AllDataPrompts.prompt_to_write_a_bio()
                            elif round_idx == 1:
                                messages = self.AllDataPrompts.prompt_to_find_a_hobby(previous_response_bio)
                            elif round_idx == 2:
                                messages = self.AllDataPrompts.prompt_to_find_a_irrelevant_hobby()
                            elif round_idx == 3:
                                messages = self.AllDataPrompts.prompt_to_create_linda_problems_original(previous_response_bio, previous_response_hobby)
                            else:
                                messages = self.AllDataPrompts.prompt_to_create_linda_problems_original_irrelevant(previous_response_bio, previous_response_hobby_irrelevant)

                elif self.args['datasets']['fallacy_type'] == 'sets':
                    if round_idx == 0:
                        messages = self.AllDataPrompts.prompt_to_write_a_syllogism()
                    elif round_idx == 1:
                        messages = self.AllDataPrompts.prompt_to_reframe_the_problem(previous_response_syllogism)
                    else:
                        messages = self.AllDataPrompts.prompt_to_reframe_the_problem_control(previous_response_syllogism, previous_response_framing_gold)
                else:
                    assert False, "Invalid fallacy type."
            else:
                raise ValueError(f'Invalid step: {step}')
            ########################################################################################

            ############################### INTERFACE OF DIFFERENT LLMS ############################
            if step == 'extract_answer':
                # always use GPT-4, because we are not evaluating llm's performance but leverage it to extract the answer as a tool
                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model='gpt-4-turbo',
                    messages=messages,
                    max_tokens=500
                )
                response = response.choices[0].message.content
            else:
                # Call Google Gemini API for Gemini models
                if re.search(r'gemini', llm_model) is not None:
                    location = "us-central1"
                    vertexai.init(project=self.project_id, location=location)
                    model = GenerativeModel(llm_model)

                    prompt = ' '.join(msg['content'] for msg in messages)
                    if re.search(r'1.5', llm_model) is not None:  # it supports vision though it is not used in this project
                        response = model.generate_content([prompt]).text
                    else:
                        chat = model.start_chat()
                        response = chat.send_message(prompt).text

                # Call Meta Llama API for Llama models
                elif re.search(r'llama', llm_model) is not None:
                    prompt = ' '.join(msg['content'] for msg in messages)
                    response = ""
                    for event in replicate.stream(
                        "meta/" + llm_model,
                        input={
                            "prompt": prompt,
                            "max_length": 1000,
                            "max_new_tokens": 500
                        },
                    ):
                        response += str(event)

                # Call Anthropic Claude API for Claude models
                elif re.search(r'claude', llm_model) is not None:
                    client = anthropic.Anthropic(
                        # defaults to os.environ.get("ANTHROPIC_API_KEY")
                        api_key=self.claude_key,
                    )

                    response = client.messages.create(
                        model=llm_model,
                        max_tokens=500,
                        system=messages[0]['content'],
                        messages=messages[1:]
                    ).content[0].text

                # Call Mistral API for Mistral models
                elif re.search(r'mistral', llm_model) is not None:
                    client = MistralClient(api_key=self.mistral_key)

                    prompt = ' '.join(msg['content'] for msg in messages)
                    response = client.chat(
                        model=llm_model,
                        messages=[ChatMessage(role="user", content=prompt)]
                    ).choices[0].message.content

                # Call OpenAI API for GPT models by default
                else:
                    client = OpenAI(api_key=self.api_key)
                    response = client.chat.completions.create(
                        model=llm_model,  # 'gpt-3.5-turbo' or 'gpt-4-turbo'
                        messages=messages,
                        max_tokens=500
                    )
                    response = response.choices[0].message.content
            ########################################################################################

            ############################### DATA GENERATION ########################################
            # record variables useful for upcoming rounds
            if step == 'generate_data' and self.args['datasets']['generate_mode'] != 'baseline':
                if self.args['datasets']['fallacy_type'] == 'linda':
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
                            linda_problem_gold = self.AllDataPrompts.random_roc_story + "\nWhich is more likely?\n(a) " + previous_response_extension \
                                                 + "\n(b) " + previous_response_extension[:-1] + " " + connector + response
                        else:
                            response = ' ' + response if response[0] != ' ' else response
                            linda_problem_random = self.AllDataPrompts.random_roc_story + "\nWhich is more likely?\n(a) " + previous_response_extension \
                                                   + "\n(b) " + previous_response_extension[:-1] + " " + connector + response

                    elif linda_problem_variant == 'variant_two':
                        if round_idx == 0:
                            previous_response_completion = response
                            response = ' ' + response if response[0] != ' ' else response
                            linda_problem_gold = self.AllDataPrompts.random_news_before_last_sentence + "\nWhich is more likely?\n(a) " + self.AllDataPrompts.random_news_last_sentence \
                                                 + "\n(b) " + self.AllDataPrompts.random_news_last_sentence[:-1] + " " + connector + response
                        else:
                            response = ' ' + response if response[0] != ' ' else response
                            linda_problem_random = self.AllDataPrompts.random_news_before_last_sentence + "\nWhich is more likely?\n(a) " + self.AllDataPrompts.random_news_last_sentence \
                                                   + "\n(b) " + self.AllDataPrompts.random_news_last_sentence[:-1] + " " + connector + response

                    elif linda_problem_variant == 'variant_three':
                        if round_idx == 0:
                            linda_problem_gold = response
                        else:
                            linda_problem_random = response

                    elif linda_problem_variant == 'variant_four':
                        if round_idx == 0:
                            linda_problem_gold = self.AllDataPrompts.parse_celebrity_few_shot(response)
                        else:
                            new_question_random_name = response
                        '''
                        if round_idx == 0:
                            previous_response_event = response
                        elif round_idx == 1:
                            previous_response_achievement = response
                        elif round_idx == 2:
                            previous_response_failure = response
                        elif round_idx == 3:
                            previous_response_problem = response
                            linda_problem_gold = response
                        elif round_idx == 4:
                            linda_problem_random = response
                        else:
                            linda_problem_random_nobody = response
                        '''

                    elif linda_problem_variant == 'variant_five':
                        if round_idx == 0:
                            previous_response_disaster = response
                        elif round_idx == 1:
                            previous_response_disaster_related = response
                        elif round_idx == 2:
                            previous_response_problem = response
                            linda_problem_gold = response
                        else:
                            linda_problem_random = response

                    elif linda_problem_variant == 'variant_six':
                        linda_problem_gold = response + self.AllDataPrompts.variant_six_suffix()
                        linda_problem_random = response + self.AllDataPrompts.variant_six_suffix_baseline()
                        correct_anwswer = self.AllDataPrompts.correct_option
                        correct_answer_baseline = self.AllDataPrompts.correct_option_baseline

                elif self.args['datasets']['fallacy_type'] == 'sets':
                    if round_idx == 0:
                        previous_response_syllogism = "All " + self.AllDataPrompts.random_object + " are " + response
                        problem_gold = "All " + self.AllDataPrompts.random_object + " are " + response

                        problem_control = problem_gold.replace("Some", "A subset of")
                        problem_control = problem_control.replace("some", "a subset of")
                        problem_control = problem_control.replace("All ", "")
                        problem_control = problem_control.capitalize()

                        problem_gold = "Is this logically sound?\n" + problem_gold
                        problem_control = "Is this logically sound?\n" + problem_control
                    elif round_idx == 1:
                        previous_response_framing_gold = response
                        problem_framing_gold = "Is this logically sound?\n" + response
                    else:
                        problem_framing_control = "Is this logically sound?\n" + response
            ########################################################################################

            # except:
            #     response = "Invalid response. "
            if self.args['datasets']['fallacy_type'] != 'sets' and verbose:
                if linda_problem_variant == 'variant_six' and self.args['datasets']['generate_mode'] != 'baseline':
                    print(f'LLM Response: {linda_problem_gold}\nbaseline {linda_problem_random}')
                else:
                    print(f'LLM Response: {response}')

        if self.args['datasets']['fallacy_type'] == 'sets' and verbose:
            print("LLM Response Problem_gold: " + problem_gold + '\n\n' + "Problem_control: " + problem_control + '\n\n' +
                  "Problem_framing_gold: " + problem_framing_gold + '\n\n' + "Problem_framing_control: " + problem_framing_control + '\n')

        if self.args['datasets']['fallacy_type'] == 'linda':
            if self.args['datasets']['generate_mode'] == 'baseline':
                return response
            else:
                if linda_problem_variant == 'variant_four':
                    #return linda_problem_gold, linda_problem_random, linda_problem_random_nobody
                    return linda_problem_gold, new_question_random_name
                elif linda_problem_variant == 'variant_six':
                    return linda_problem_gold, linda_problem_random, correct_anwswer, correct_answer_baseline
                else:
                    return linda_problem_gold, linda_problem_random
        elif self.args['datasets']['fallacy_type'] == 'sets':
            return problem_gold, problem_control, problem_framing_gold, problem_framing_control
        else:
            assert False, "Invalid fallacy type."
