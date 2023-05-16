#!/usr/bin/env python3
#SBATCH --job-name mice-generation
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/logs/generation/%A.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/logs/generation/%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 6
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --time 10
#SBATCH --requeue

import os
import sys
from math import sqrt
import argparse
from collections import defaultdict
from datetime import datetime
import random

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

sys.path.append("/coc/pskynet6/dhe83/mice/src")
from utils import *
import config

def similarity_scores(train_data, test_data, dataset, encoder_model):

    model = SentenceTransformer(encoder_model)
    device = torch.device("cuda")
    model.to(device)

    similarity_map = defaultdict(dict)

    test_embeddings = {}
    for test_example in test_data:
        test_prompt = format_example(test_example, dataset)
        test_embeddings[test_example['idx']] = torch.tensor(model.encode(test_prompt))

    train_embeddings = {}
    for train_example in train_data:
        train_prompt = train_example['in_context']
        train_embeddings[train_example['idx']] = torch.tensor(model.encode(train_prompt))

    similarity = nn.CosineSimilarity(dim=1)
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

def bayesian_noise_reduction(in_context: int, max_num_prompts: int, model_name: str) -> dict(list()):
    #TODO: do this
    print("hi")

def main():
    parser = argparse.ArgumentParser(description='Generate json dictionary consisting of test_idx: train_indices)')
    parser.add_argument('--dataset', choices=config.tasks)
    parser.add_argument('--experiment_id', type=int)
    parser.add_argument('--in_context', default=2, type=int)
    parser.add_argument('--max_num_prompts', default=1, type=int)
#   parser.add_argument('--uuid', type=str)

    args = parser.parse_args()
    dataset, exp_id, in_context, max_num_prompts = args.dataset, args.experiment_id, args.in_context, args.max_num_prompts
    assert None not in [dataset, exp_id, in_context, max_num_prompts]

#   experiment_id = get_log_with_uuid(args.uuid)['experiment_id'] if args.uuid else args.experiment_id

#   exp_info = get_experiment_info(experiment_id)
    exp_dir = os.path.join(config.experiments, dataset)
    found = False
    for d in os.listdir(exp_dir):
        idx = int(d.split(config.delim)[0])
        if idx == exp_id:
            found = True
            exp_dir = os.path.join(exp_dir,d)

    assert found

    generation_dir = os.path.join(exp_dir, f"{args.in_context}_{args.max_num_prompts}")

    train_data = read_jsonl(os.path.join(exp_dir, 'train.jsonl'))
    test_data = read_jsonl(os.path.join(exp_dir, 'test.jsonl'))

    similarity_map = similarity_scores(train_data, test_data, dataset, "all-roberta-large-v1")
    prompt_map = similar_generator(similarity_map, in_context, max_num_prompts)

    os.makedirs(generation_dir, exist_ok=True)

    print("Writing similarity scores...", end="")
    write_json(similarity_map, os.path.join(generation_dir, 'similarity_scores.json'))
    print("done!")

    print("Writing prompt map...", end="")
    write_json(prompt_map, os.path.join(generation_dir, 'prompt_map.json'))
    print("done!")

#   print("Logging info...", end="")
#   info = os.path.join(exp_info['location'], 'info.json')
#   info_data = read_json(info)
#   generation_id = max([int(id) for id in info_data['generations'].keys()], default=0) + 1
#   info_data['generations'][generation_id] = {'created': str(datetime.now()),
#                                               'location': generation_dir,
#                                               'ordering': args.ordering,
#                                               'in_context': args.in_context,
#                                               'max_num_prompts': args.max_num_prompts,
#                                               'encoder': args.encoder}
#   write_json(info_data, info)
#   print("done!")

#   if args.uuid:
#       log_file = os.path.join(config.logs, f"{args.uuid}.json")
#       log = read_json(log_file)
#       log['generation_id'] = str(generation_id)
#       log['last_modified'] = str(datetime.now())
#       log['status'] = "inference"
#       write_json(log, log_file)

if __name__ == '__main__':
    main()
