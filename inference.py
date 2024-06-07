from tqdm import tqdm
import os
import numpy as np
import re

from query_llm import QueryLLM
from utils import *


def grade_model_answer(model_answer, target_answer, incorrect_answers, LLM, grader, args):
    if args['datasets']['fallacy_type'] == 'linda':
        # check if we need to summarize the final answer from the chain of thought reasoning
        if re.search(r'\(a\)', model_answer) is not None and re.search(r'\(b\)', model_answer) is not None:
            if args['inference']['verbose']:
                print(f"{Colors.OKGREEN}{'Summarizing the final answer from the chain of thought reasoning...'}{Colors.ENDC}")
            model_answer_summary = LLM.query_llm(model_answer=model_answer, llm_model=args['models']['llm_model'], step='extract_answer', verbose=args['inference']['verbose'])
            if re.search(r'\(a\)', model_answer_summary) is not None and re.search(r'\(b\)', model_answer_summary) is not None:
                model_answer_summary = "Failed to answer."
        else:
            model_answer_summary = model_answer

        candidate_options = [r'\(a\)', r'\(b\)'] if len(incorrect_answers) == 1 else [r'\(a\)', r'\(b\)', r'\(c\)']
        matched = False
        for option in candidate_options:
            if re.search(option, target_answer) is not None and re.search(option, model_answer_summary) is not None:
                init_grades = ['[Correct]']
                matched = True
                break
        if not matched:
            init_grades = ['[Incorrect]']
        grader.accumulate_grades(args, init_grades)

    elif args['datasets']['fallacy_type'] == 'sets':
        # target answer is always no in this case
        no_pattern = r'\b(no)\b[.,!?]?'
        yes_pattern = r'\b(yes)\b[.,!?]?'
        if re.search(no_pattern, model_answer.lower()) is not None and re.search(yes_pattern, model_answer.lower()) is not None:
            if args['inference']['verbose']:
                print(f"{Colors.OKGREEN}{'Summarizing the final answer from the chain of thought reasoning...'}{Colors.ENDC}")
            model_answer_summary = LLM.query_llm(model_answer=model_answer, llm_model=args['models']['llm_model'], step='extract_answer', verbose=args['inference']['verbose'])
            if re.search(no_pattern, model_answer.lower()) is not None and re.search(yes_pattern, model_answer.lower()) is not None:
                model_answer_summary = "Failed to answer."
        else:
            model_answer_summary = model_answer

        if re.search(no_pattern, model_answer_summary.lower()) is not None:
            init_grades = ['[Correct]']
        else:
            init_grades = ['[Incorrect]']
        grader.accumulate_grades(args, init_grades)
    else:
        raise ValueError(f"Invalid fallacy type: {args['datasets']['fallacy_type']}")

    return init_grades


def inference(device, args, test_loader):
    LLM = QueryLLM(args)
    grader = [Grader(), Grader()]  # Graders for initial model answer and retry model answer

    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            if data == -1:
                continue  # Skip the None entries
            question_id, question, target_answer, generation_mode \
                = data['question_id'][0], data['question'][0], data['target_answer'][0], data['generation_mode'][0]
            
            incorrect_answers = data['incorrect_answers']
            if len(incorrect_answers) == 1:
                incorrect_answers = incorrect_answers[0]
            else:
                incorrect_answers = [answer[0] for answer in incorrect_answers]

            ########### Answer the question ###########
            init_model_answer = LLM.query_llm(question=question, llm_model=args['models']['llm_model'], step='answer_question', verbose=args['inference']['verbose'])

            ########### Multi-agent process ###########
            # We haven't yet implemented the multi-agent process for the retry model answer
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
            init_grades = grade_model_answer(init_model_answer, target_answer, incorrect_answers, LLM, grader[0], args)
            if retry:
                retry_grades = grade_model_answer(retry_model_answer, target_answer, incorrect_answers, LLM, grader[1], args)

            ########### Record responses ###########
            response_dict = {'question_id': str(question_id.item()), 'question': question, 'target_answer': target_answer, 'retry': retry, 'use_multi_agents': args['inference']['use_multi_agents'],
                             'init_model_answer': init_model_answer, 'init_grades': init_grades} #, 'init_answer_majority_vote': init_answer_majority_vote}
            if retry:
                response_dict['retry_model_answer'] = retry_model_answer
                response_dict['retry_grades'] = retry_grades
                # response_dict['retry_answer_majority_vote'] = retry_answer_majority_vote

            if (batch_count + 1) % args['inference']['print_every'] == 0 or (batch_count + 1) == len(test_loader):
                print_response(retry, grader, batch_count, len(test_loader), args['inference']['output_dir'], llm_model=args['models']['llm_model'], data_file=args['datasets']['file_name'], eval_mode=args['inference']['mode'])

            if args['inference']['save_output_response']:
                write_response_to_json(question_id, response_dict, args['inference']['output_dir'], llm_model=args['models']['llm_model'], data_file=args['datasets']['file_name'], eval_mode=args['inference']['mode'])
