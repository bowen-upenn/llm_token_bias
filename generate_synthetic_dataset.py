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
            new_question_gold, new_question_random = LLM.query_llm(llm_model=args['models']['llm_model'], step='generate_data', verbose=args['inference']['verbose'])
            new_questions = [new_question_gold, new_question_random]

            try:
                for i, curr_new_question in enumerate(new_questions):
                    # the shorter answer in Linda Problem is the target answer
                    pattern_a = r"\(a\)\s*(.*?)\s*(?=\(b\))"
                    pattern_b = r"\(b\)\s*(.*)"

                    choice_a = re.findall(pattern_a, curr_new_question)
                    choice_b = re.findall(pattern_b, curr_new_question)

                    new_target_answer = choice_a[0] if len(choice_a[0]) < len(choice_b[0]) else choice_b[0]

                    ########### Record New Data Entry ###########
                    generation_mode = 'gold' if i == 0 else 'random'
                    logical_connector = args['datasets']['connector']
                    response_dict = {'question_idx': n, 'question': curr_new_question, 'target_answer': new_target_answer, 'generation_mode': generation_mode}
                    write_response_to_json(n, response_dict, synthetic_data_filename, fallacy_type=fallacy_type,
                                           generation_mode=generation_mode, linda_problem_variant=linda_problem_variant, logical_connector=logical_connector)
            except:
                continue
