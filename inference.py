from tqdm import tqdm
import os
import numpy as np
import re

from query_llm import QueryLLM
from utils import *


def inference(device, args, test_loader):
    LLM = QueryLLM(args)
    grader = [Grader(), Grader()]  # Graders for initial model answer and retry model answer

    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            question_id, question, target_answer, incorrect_answers, generation_mode \
                = data['question_id'][0], data['question'][0], data['target_answer'][0], data['incorrect_answers'][0], data['generation_mode'][0]

            ########### Answer the question ###########
            init_model_answer = LLM.query_llm(question=question, llm_model=args['models']['llm_model'], step='answer_question', verbose=args['inference']['verbose'])

            ########### Multi-agent process ###########
            retry = False
            if args['inference']['use_multi_agents']:
                critic = LLM.query_llm(question=question, llm_model=args['models']['llm_model'], step='critic_answer', model_answer=init_model_answer, verbose=args['inference']['verbose'])
                if re.search(r'\[Yes\]', critic) is not None:
                    retry = True
                    retry_model_answer = LLM.query_llm(question=question, llm_model=args['models']['llm_model'], step='reanswer_question', model_answer=init_model_answer,
                                                       critic=critic, verbose=args['inference']['verbose'])
                else:
                    retry_model_answer = init_model_answer
            else:
                retry_model_answer = init_model_answer

            ########### Grade model answers ###########
            candidate_options = [r'\(a\)', r'\(b\)'] if len(incorrect_answers) == 1 else [r'\(a\)', r'\(b\)', r'\(c\)']
            matched = False
            for option in candidate_options:
                if re.search(option, target_answer) is not None and re.search(option, init_model_answer) is not None:
                    init_grades = ['[Correct]']
                    matched = True
                    break
            if not matched:
                init_grades = ['[Incorrect]']
            grader[0].accumulate_grades(args, init_grades)

            if retry:
                matched = False
                for option in candidate_options:
                    if re.search(option, target_answer) is not None and re.search(option, retry_model_answer) is not None:
                        retry_grades = ['[Correct]']
                        matched = True
                        break
                if not matched:
                    retry_grades = ['[Incorrect]']
                grader[1].accumulate_grades(args, retry_grades)

            # # If using LLM graders
            # init_grades, retry_grades = [], []
            # for grader_id in range(args['inference']['num_graders']):
            #     init_grades.append(LLM.query_llm(question=question, llm_model=args['models']['llm_model'], step='grade_answer', target_answer=target_answer, model_answer=init_model_answer,
            #                                      grader_id=grader_id, verbose=args['inference']['verbose']))
            #     if retry:
            #         retry_grades.append(LLM.query_llm(question=question, llm_model=args['models']['llm_model'], step='grade_answer', target_answer=target_answer, model_answer=retry_model_answer,
            #                                           grader_id=grader_id, verbose=args['inference']['verbose']))
            #
            # init_answer_majority_vote = grader[0].accumulate_grades(args, init_grades)
            # if retry:
            #     retry_answer_majority_vote = grader[1].accumulate_grades(args, retry_grades)

            ########### Record responses ###########
            response_dict = {'question_id': str(question_id.item()), 'question': question, 'target_answer': target_answer, 'retry': retry, 'use_multi_agents': args['inference']['use_multi_agents'],
                             'init_model_answer': init_model_answer, 'init_grades': init_grades} #, 'init_answer_majority_vote': init_answer_majority_vote}
            if retry:
                response_dict['retry_model_answer'] = retry_model_answer
                response_dict['retry_grades'] = retry_grades
                # response_dict['retry_answer_majority_vote'] = retry_answer_majority_vote

            if (batch_count + 1) % args['inference']['print_every'] == 0 or (batch_count + 1) == len(test_loader):
                print_response(retry, grader, batch_count, len(test_loader), args['inference']['output_response_filename'], data_file=args['datasets']['file_name'], eval_mode=args['inference']['mode'])

            if args['inference']['save_output_response']:
                write_response_to_json(question_id, response_dict, args['inference']['output_response_filename'], data_file=args['datasets']['file_name'], eval_mode=args['inference']['mode'])
