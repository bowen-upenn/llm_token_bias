from tqdm import tqdm
import os
import numpy as np
import re

from query_llm import QueryLLM
from utils import *


def inference(device, args, test_loader):
    LLM = QueryLLM(args)
    grader = [Grader(), Grader()]  # Graders for initial model answer and retry model answer
    output_response_filename = args['inference']['output_response_filename']

    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            question_id, question, target_answer = data['question_id'][0], data['question'][0], data['target_answer'][0]

            ########### Answer the question ###########
            init_model_answer = LLM.query_llm(question=question, llm_model=args['llm']['llm_model'], step='answer_question', verbose=args['inference']['verbose'])

            ########### Multi-agent process ###########
            retry = False
            if args['inference']['use_multi_agents']:
                critic = LLM.query_llm(question=question, llm_model=args['llm']['llm_model'], step='critic_answer', model_answer=init_model_answer, verbose=args['inference']['verbose'])
                if re.search(r'\[Yes\]', critic) is not None:
                    retry = True
                    retry_model_answer = LLM.query_llm(question=question, llm_model=args['llm']['llm_model'], step='reanswer_question', model_answer=init_model_answer,
                                                       critic=critic, verbose=args['inference']['verbose'])
                else:
                    retry_model_answer = init_model_answer
            else:
                retry_model_answer = init_model_answer

            ########### Grade model answers ###########
            init_grades, retry_grades = [], []
            for grader_id in range(args['inference']['num_graders']):
                init_grades.append(LLM.query_llm(question=question, llm_model=args['llm']['llm_model'], step='grade_answer', target_answer=target_answer, model_answer=init_model_answer,
                                                 grader_id=grader_id, verbose=args['inference']['verbose']))
                if retry:
                    retry_grades.append(LLM.query_llm(question=question, llm_model=args['llm']['llm_model'], step='grade_answer', target_answer=target_answer, model_answer=retry_model_answer,
                                                      grader_id=grader_id, verbose=args['inference']['verbose']))

            init_answer_majority_vote = grader[0].accumulate_grades(args, init_grades)
            if retry:
                retry_answer_majority_vote = grader[1].accumulate_grades(args, retry_grades)

            ########### Record responses ###########
            response_dict = {'question_id': str(question_id.item()), 'question': question, 'target_answer': target_answer, 'retry': retry, 'use_multi_agents': args['inference']['use_multi_agents'],
                             'init_model_answer': init_model_answer, 'init_grades': init_grades, 'init_answer_majority_vote': init_answer_majority_vote}
            if retry:
                response_dict['retry_model_answer'] = retry_model_answer
                response_dict['retry_grades'] = retry_grades
                response_dict['retry_answer_majority_vote'] = retry_answer_majority_vote

            if (batch_count + 1) % args['inference']['print_every'] == 0 or (batch_count + 1) == len(test_loader):
                print_response(retry, grader, batch_count, len_test_loader, output_response_filename)

            if args['inference']['save_output_response']:
                write_response_to_json(question_id, response_dict, output_response_filename)
