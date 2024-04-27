import os
import json
import torch
from torch.utils.data import Dataset
from utils import *


class FallacyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        with open(os.path.join(self.args['datasets']['data_dir'], self.args['datasets']['file_name']), 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        idx_key = str(idx.item())
        if idx_key not in self.annotations:
            return -1  # Return None if the idx does not exist in annotations

        annot = self.annotations[idx_key]
        question_id = annot['question_idx']
        question = annot['question']
        target_answer = annot['target_answer']
        incorrect_answers = annot['incorrect_answer']
        generation_mode = annot['generation_mode']

        if self.args['inference']['verbose']:
            # curr_data = 'question_id: ' + str(question_id) + ' question: ' + question + ' target_answer: ' + target_answer + 'generation_mode: ' + generation_mode
            print('Question', question, 'Target Answer', target_answer)
            # print(f'{Colors.HEADER}{curr_data}{Colors.ENDC}')

        return {'question_id': question_id, 'question': question, 'target_answer': target_answer, 'incorrect_answers': incorrect_answers, 'generation_mode': generation_mode}
