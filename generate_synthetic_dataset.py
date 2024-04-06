from tqdm import tqdm
import os
import numpy as np
import re

from query_llm import QueryLLM
from utils import *


def data_generation(device, args):
    synthetic_data_filename = args['datasets']['synthetic_data_filename']
    fallacy_type = args['datasets']['fallacy_type']
    generate_mode = args['datasets']['generate_mode']
    linda_problem_variant = args['datasets']['linda_problem_variant']
    LLM = QueryLLM(args)

    with torch.no_grad():
        ########### In-Context Learning ###########
        for icl_idx in tqdm(range(args['datasets']['num_synthetic_examples'])):
            new_question = LLM.query_llm(llm_model=args['models']['llm_model'], step='generate_data', verbose=args['inference']['verbose'])

            # the shorter answer in Linda Problem is the target answer
            pattern_a = r"\(a\)\s*(.*?)\s*(?=\(b\))"
            pattern_b = r"\(b\)\s*(.*)"

            choice_a = re.findall(pattern_a, new_question)
            choice_b = re.findall(pattern_b, new_question)

            new_target_answer = choice_a[0] if len(choice_a[0]) < len(choice_b[0]) else choice_b[0]

            ########### Record New Data Entry ###########
            response_dict = {'icl_idx': icl_idx, 'question': new_question, 'target_answer': new_target_answer}
            write_response_to_json(icl_idx, response_dict, synthetic_data_filename,
                                   fallacy_type=fallacy_type, generate_mode=generate_mode, linda_problem_variant=linda_problem_variant)
