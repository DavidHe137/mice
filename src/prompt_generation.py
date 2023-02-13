import os
from pathlib import Path
import argparse
from utils import *
from templates import format_example

from transformers import AutoTokenizer
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer

import torch
import torch.nn as nn
from collections import defaultdict

import random
def similarity_scores(encoder_model):

    model = SentenceTransformer(encoder_model)
    device = torch.device("cpu") #FIXME: cuda
    model.to(device)

    global train_data, test_data, dataset
    similarity_map = defaultdict({})

    test_embeddings = {}
    for test_example in test_data:
        test_prompt = format_example(test_example, dataset, includeLabel=False)
        test_embeddings[test_example['idx']] = torch.tensor(model.encode(test_prompt))

    train_embeddings = {}
    for train_example in train_data:
        train_prompt = format_example(train_example, dataset, includeLabel=True)
        train_embeddings[train_example['idx']] = torch.tensor(model.encode(test_prompt))
    
    print(train_embeddings)
    #similarity_map = {test_example['idx'], {train_example['idx'], nn.CosineSimilarity() for train_example in train_data}  for test_example in test_data}

def random_sampling(in_context: int, max_num_prompts: int) -> dict(list()):
    global train_data, test_data, dataset
    prompt_map = {}
    #for test_example in test_data:
        #ids = sample 


def bayesian_noise_reduction(in_context: int, max_num_prompts: int) -> dict(list()):
    print("hi")

def main():
    parser = argparse.ArgumentParser(description='Configure dataset, model-size, method, data splits.')
    parser.add_argument('experiment_id', type=int)
    parser.add_argument('--similarity_encoder', default='all-roberta-large-v1', type=str)

    args = parser.parse_args()

    # gather absolute paths
    project_root = Path(__file__).resolve().parents[1]
    exp_home = os.path.join(project_root, 'experiments')

    exp_summary = os.path.join(exp_home, 'summary.json') 

    exp_info = {}
    # check to see experiment id exists
    if os.path.exists(exp_summary):        
        exp_summary_data = read_json(exp_summary)['summary']

        exp_info = {}
        for exp in exp_summary_data:
            if exp['id'] == args.experiment_id:
                exp_info = read_json(os.path.join(exp['location'], 'info.json'))
        
        assert exp_info
    
    global train_data, test_data, dataset
    train_data = read_jsonl(os.path.join(exp_info['location'], 'train.jsonl'))
    test_data = read_jsonl(os.path.join(exp_info['location'], 'test.jsonl'))
    dataset = exp['dataset']

    similarity_scores(args.similarity_encoder)
#    prompt_map = random_sampling(train, test, exp_info['in_context'], exp_info['max_num_prompts'])

#    write_json(prompt_map, os.path.join(exp_info['location'], 'prompt_map.json'))
              
if __name__ == '__main__':
    main()
    #generate prompt_map

    #called in bash
    #run inference inside experiment folder
    #log inference flops and time

    #combine prompts using mice_sampling, majority vote

    #evaluate using metrics (superGLUE, huggingface evaluate, replicated)