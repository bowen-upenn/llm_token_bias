from tqdm import tqdm
import os
import numpy as np
import re

from query_llm import QueryLLM
from utils import *


def data_generation(device, args):
    synthetic_data_filename = args['datasets']['synthetic_data_filename']
    fallacy_type = args['datasets']['fallacy_type']
    linda_problem_variant = args['datasets']['linda_problem_variant']
    LLM = QueryLLM(args)

    with torch.no_grad():
        ########### In-Context Learning ###########
        for n in tqdm(range(args['datasets']['num_synthetic_examples'])):
            if args['datasets']['generate_mode'] == 'baseline':
                new_question_baseline = LLM.query_llm(llm_model=args['models']['llm_model'], step='generate_data', verbose=args['inference']['verbose'])
                new_questions = [new_question_baseline]
            else:
                if linda_problem_variant == 'variant_six':
                    new_question_gold, new_question_baseline, correct_answer, correct_answer_baseline = LLM.query_llm(llm_model=args['models']['llm_model'], step='generate_data', verbose=args['inference']['verbose'])
                    new_questions = [new_question_gold, new_question_baseline]
                #elif args['datasets']['linda_problem_variant'] == 'variant_four':
                    #new_question_gold, new_question_random_achievement, new_question_random_name = LLM.query_llm(llm_model=args['models']['llm_model'], step='generate_data', verbose=args['inference']['verbose'])
                    #new_questions = [new_question_gold, new_question_random_achievement, new_question_random_name]
                else:
                    new_question_gold, new_question_random = LLM.query_llm(llm_model=args['models']['llm_model'], step='generate_data', verbose=args['inference']['verbose'])
                    new_questions = [new_question_gold, new_question_random]

            try:
                for i, curr_new_question in enumerate(new_questions):
                    # the shorter answer in Linda Problem is the target answer
                    pattern_a = r"\(a\)\s*(.*?)\s*(?=\(b\))"
                    choice_a = re.findall(pattern_a, curr_new_question)
                    if choice_a[0][-1] == '.':
                        choice_a[0] = choice_a[0][:-1]
                    choice_a[0] = '(a) ' + choice_a[0]

                    if args['datasets']['linda_problem_variant'] == 'variant_six':
                        pattern_b = r"\(b\)\s*(.*)\s*(?=\(c\))"
                        pattern_c = r"\(c\)\s*(.*)"
                        choice_b = re.findall(pattern_b, curr_new_question)
                        choice_c = re.findall(pattern_c, curr_new_question)
                        if choice_b[0][-1] == '.':
                            choice_b[0] = choice_b[0][:-1]
                        if choice_c[0][-1] == '.':
                            choice_c[0] = choice_c[0][:-1]
                        choice_b[0] = '(b) ' + choice_b[0]
                        choice_c[0] = '(c) ' + choice_c[0]

                        if i == 0: # golden
                            new_target_answer = choice_a[0] if correct_answer == 0 else choice_b[0] if correct_answer == 1 else choice_c[0]
                            incorrect_answer = [choice_b[0], choice_c[0]] if correct_answer == 0 else [choice_a[0], choice_c[0]] if correct_answer == 1 else [choice_a[0], choice_b[0]]
                        else: # baseline
                            new_target_answer = choice_a[0] if correct_answer_baseline == 0 else choice_b[0] if correct_answer_baseline == 1 else choice_c[0]
                            incorrect_answer = [choice_b[0], choice_c[0]] if correct_answer_baseline == 0 else [choice_a[0], choice_c[0]] if correct_answer_baseline == 1 else [choice_a[0], choice_b[0]]
                    else:
                        pattern_b = r"\(b\)\s*(.*)"
                        choice_b = re.findall(pattern_b, curr_new_question)
                        if choice_b[0][-1] == '.':
                            choice_b[0] = choice_b[0][:-1]
                        choice_b[0] = '(b) ' + choice_b[0]

                        new_target_answer = choice_a[0] if len(choice_a[0]) < len(choice_b[0]) else choice_b[0]
                        incorrect_answer = choice_a[0] if new_target_answer == choice_b[0] else choice_b[0]

                    ########### Record New Data Entry ###########
                    if args['datasets']['generate_mode'] == 'baseline':
                        generation_mode = 'baseline'
                    else:
                        generation_mode = 'gold' if i == 0 else 'random'
                        #if args['datasets']['linda_problem_variant'] == 'variant_four' and i != 0:
                        #    generation_mode += '_achievement' if i == 1 else '_name'
                    logical_connector = args['datasets']['connector']
                    response_dict = {'question_idx': n, 'question': curr_new_question, 'target_answer': new_target_answer, 'incorrect_answer': incorrect_answer, 'generation_mode': generation_mode}
                    write_response_to_json(n, response_dict, synthetic_data_filename, fallacy_type=fallacy_type,
                                           generation_mode=generation_mode, linda_problem_variant=linda_problem_variant, logical_connector=logical_connector)
            except:
                continue
