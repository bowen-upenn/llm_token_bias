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
from dataloader import *


if __name__ == "__main__":
    print('Torch', torch.__version__, 'Torchvision', torchvision.__version__)
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--model', type=str, default="gpt4", help='Set LLM model (gpt-3.5-turbo, gpt-4-turbo-preview)')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    parser.add_argument('--multi_agent', dest='verbose', action='store_true', help='Set use multi-agents to True')
    cmd_args = parser.parse_args()
    if cmd_args.model == "gpt3.5":
        cmd_args.model = "gpt-3.5-turbo"
    if cmd_args.model == "gpt4":
        cmd_args.model = "gpt-4-turbo-preview"

    # Override args from config.yaml with command-line arguments if provided
    args['models']['llm_model'] = cmd_args.model if cmd_args.verbose is not None else args['inference']['verbose']
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']
    args['inference']['use_multi_agents'] = cmd_args.verbose if cmd_args.multi_agent is not None else args['inference']['use_multi_agents']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = torch.cuda.device_count()
    assert world_size == 1
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Prepare datasets
    print("Loading the datasets...")
    test_dataset = FallacyDataset(args)

    torch.manual_seed(0)
    if args['datasets']['test_on_subset']:
        test_subset_idx = torch.randperm(len(test_dataset))[:num_test_data]
    else:
        test_subset_idx = torch.randperm(len(test_dataset))
    test_subset = Subset(test_dataset, test_subset_idx)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    print('num of train, test:', 0, len(test_subset))

    # Start inference
    print(args)
    inference(device, args, test_loader)
