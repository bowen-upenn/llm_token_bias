import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import yaml
import os
import json
import argparse

from utils import *
from inference import inference
from generate_synthetic_dataset import data_generation
from dataloader import *


if __name__ == "__main__":
    print('Torch', torch.__version__, 'Torchvision', torchvision.__version__)
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args['models']['credential_path']

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--model', type=str, default="gpt4", help='Set LLM model (gpt-3.5-turbo, gpt-4-turbo-preview)')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    parser.add_argument('--multi_agent', dest='multi_agent', action='store_true', help='Set use multi-agents to True')
    parser.add_argument('--task', type=str, default="task", help='Set task (inference, data)')
    parser.add_argument('--fallacy', type=str, default="linda", help='Set logical fallacy type (linda)')
    parser.add_argument('--gen_mode', type=str, default="baseline", help='Set generate mode for synthetic dataset (baseline, gold, random)')
    cmd_args = parser.parse_args()
    if cmd_args.model == "gpt3.5":
        cmd_args.model = "gpt-3.5-turbo"
    if cmd_args.model == "gpt4":
        cmd_args.model = "gpt-4-turbo-preview"

    # Override args from config.yaml with command-line arguments if provided
    args['models']['llm_model'] = cmd_args.model if cmd_args.verbose is not None else args['inference']['verbose']
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']
    args['inference']['use_multi_agents'] = cmd_args.multi_agent if cmd_args.multi_agent is not None else args['inference']['use_multi_agents']
    args['inference']['task'] = cmd_args.task if cmd_args.task is not None else args['inference']['task']
    args['datasets']['fallacy_type'] = cmd_args.fallacy if cmd_args.fallacy is not None else args['datasets']['fallacy_type']
    args['datasets']['generate_mode'] = cmd_args.gen_mode if cmd_args.gen_mode is not None else args['datasets']['generate_mode']

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = torch.cuda.device_count()
    assert world_size == 1
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Start inference
    print(args)
    if args['inference']['task'] == 'inference':
        # Prepare datasets
        print("Loading the datasets...")
        test_dataset = FallacyDataset(args)

        if args['datasets']['test_on_subset']:
            test_subset_idx = torch.randperm(len(test_dataset))[:num_test_data]
        else:
            test_subset_idx = torch.randperm(len(test_dataset))
        test_subset = Subset(test_dataset, test_subset_idx)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
        print('num of train, test:', 0, len(test_subset))

        inference(device, args, test_loader)
    elif args['inference']['task'] == 'data':
        data_generation(device, args)
    else:
        print('Invalid task')
