from tqdm import tqdm
import os
import numpy as np
import re

from query_llm import QueryLLM
from utils import *


def data_generation(device, args, test_loader):
    LLM = QueryLLM(args)
    grader = [Grader(), Grader()]  # Graders for initial model answer and retry model answer
    synthetic_data_filename = args['datasets']['synthetic_data_filename']

    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            question_id, question, target_answer = data['question_id'][0], data['question'][0], data['target_answer'][0]

            ########### In-Context Learning ###########
            for icl_idx in range(args['datasets']['num_icl']):
                new_question_id = str(question_id.item()) + '_' + str(icl_idx)
                new_question = LLM.query_llm(question=question, llm_model=args['llm']['llm_model'], step='generate_data', verbose=args['inference']['verbose'])

                # the shorter answer in Linda Problem is the target answer
                pattern_a = r"\(a\)\s*(.*?)\s*(?=\(b\))"
                pattern_b = r"\(b\)\s*(.*)"

                choice_a = re.findall(pattern_a, new_question)
                choice_b = re.findall(pattern_b, new_question)

                new_target_answer = choice_a[0] if len(choice_a[0]) < len(choice_b[0]) else choice_b[0]

                ########### Record New Data Entry ###########
                response_dict = {'question_id': new_question_id, 'question': new_question, 'target_answer': new_target_answer}
                write_response_to_json(new_question_id, response_dict, synthetic_data_filename)
