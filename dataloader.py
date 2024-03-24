import os
import json
import torch
from torch.utils.data import Dataset


class FallacyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        with open(os.path.join(self.args['datasets']['data_dir'], self.args['datasets']['file_name']), 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annot = self.annotations[idx]
        question_id = annot['question_id']
        question = annot['question']
        target_answer = annot['target_answer']

        if self.args['inference']['verbose']:
            curr_data = 'question_id: ' + str(question_id) + ' question: ' + question + ' target_answer: ' + target_answer
            print(f'{Colors.HEADER}{curr_data}{Colors.ENDC}')

        return {'question_id': question_id, 'question': question, 'target_answer': target_answer}
