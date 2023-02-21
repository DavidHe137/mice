import os
from math import sqrt
import argparse
from utils import *

from sentence_transformers import SentenceTransformer

import torch
import torch.nn as nn
from collections import defaultdict
from datetime import datetime

import random
def similarity_scores(train_data, test_data, dataset, encoder_model):

    model = SentenceTransformer(encoder_model)
    device = torch.device("cuda")
    model.to(device)

    similarity_map = defaultdict(dict)

    test_embeddings = {}
    for test_example in test_data:
        test_prompt = format_example(test_example, dataset, includeLabel=False)
        test_embeddings[test_example['idx']] = torch.tensor(model.encode(test_prompt))

    train_embeddings = {}
    for train_example in train_data:
        train_prompt = format_example(train_example, dataset, includeLabel=True)
        train_embeddings[train_example['idx']] = torch.tensor(model.encode(train_prompt))

    similarity = nn.CosineSimilarity(dim=1) #TODO: possible hyperparameter
    similarity_map = {test_example['idx']:
                        sorted([(train_example['idx'], similarity(
                                                    test_embeddings[test_example['idx']].unsqueeze(0),
                                                    train_embeddings[train_example['idx']].unsqueeze(0),).item())
                                                    for train_example in train_data], key=lambda x: x[1], reverse=True)
                        for test_example in test_data}

    return similarity_map

def similar_generator(similarity_map: dict, in_context: int, max_num_prompts: int) -> dict(list()):
    prompt_map = {}
    for test_idx in similarity_map:
        #TODO: generalize for k in_context

        # then extract the top-sqrt(num_prompts) from similarity and generate all pairs
        t = int(sqrt(max_num_prompts))

        # special case for t=1: select the top-2 similar and put into a single prompt
        if t == 1:
            top_2 = list(map(lambda x: x[0], similarity_map[test_idx][:2]))
            prompt_map[test_idx] = [(top_2[0], top_2[1])]
        else:
            top_t = list(map(lambda x: x[0], similarity_map[test_idx][:t]))
            prompt_map[test_idx] = [(ex1, ex2) for ex1 in top_t for ex2 in top_t]

    return prompt_map

def random_generator(train_data, test_data, in_context: int, max_num_prompts: int) -> dict(list()):
    prompt_map = {}
    train_indices = [train_example['idx'] for train_example in train_data]
    train_idx_pairs = [(ex1, ex2) for ex1 in train_indices for ex2 in train_indices]
    num_prompts = max(len(train_idx_pairs, max_num_prompts))

    #TODO: generalize for k in_context
    for test_example in test_data:
        prompt_map[test_example['idx']] = random.sample(train_idx_pairs, num_prompts)

    return prompt_map

def bayesian_noise_reduction(in_context: int, max_num_prompts: int) -> dict(list()):
    print("hi")

def main():
    parser = argparse.ArgumentParser(description='Generate json dictionary consisting of test_idx: train_indices)')
    parser.add_argument('experiment_id', type=str)
    parser.add_argument('--method', choices=['similar', 'random', 'bayesian_noise'])
    parser.add_argument('--in_context', default=2, type=int)
    parser.add_argument('--max_num_prompts', default=1, type=int) #FIXME: default value
    parser.add_argument('--encoder', default='all-roberta-large-v1', type=str)

    #TODO: token limits, generation length
    args = parser.parse_args()

    exp_info = get_experiment_info(args.experiment_id)
    generation_dir = os.path.join(exp_info['location'], 'generations', f"{args.method}_{args.in_context}_{args.max_num_prompts}_{args.encoder}")
    if os.path.exists(generation_dir):
        print(f"{generation_dir} already exists.")
        return

    train_data = read_jsonl(os.path.join(exp_info['location'], 'train.jsonl'))
    test_data = read_jsonl(os.path.join(exp_info['location'], 'test.jsonl'))

    similarity_map = {}
    prompt_map = {}

    if args.method == 'similar':
        similarity_map = similarity_scores(train_data, test_data, exp_info['dataset'], args.encoder)
        prompt_map = similar_generator(similarity_map, args.in_context, args.max_num_prompts)
    elif args.method == 'random':
        prompt_map = random_generator(train_data, test_data, args.in_context, args.max_num_prompts)
    
    os.makedirs(generation_dir, exist_ok=True)
    
    if similarity_map:
        print("Writing similarity scores...", end="")
        write_json(similarity_map, os.path.join(generation_dir, 'similarity_scores.json'))
        print("done!")

    print("Writing prompt map...", end="")
    write_json(prompt_map, os.path.join(generation_dir, 'prompt_map.json'))
    print("done!")

    print("Logging info...", end="")
    info = os.path.join(exp_info['location'], 'info.json')
    info_data = read_json(info)
    info_data['generations'][len(info_data['generations']) + 1] = {'created': str(datetime.now()),
                                                                   'location': generation_dir,
                                                                   'method': args.method,
                                                                   'in_context': args.in_context,
                                                                   'max_num_prompts': args.max_num_prompts,
                                                                   'encoder': args.encoder}
    write_json(info_data, info)
    print("done!")

if __name__ == '__main__':
    main()
