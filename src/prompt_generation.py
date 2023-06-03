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
from itertools import product

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

sys.path.append("/coc/pskynet6/dhe83/mice/src")
from utils import *
from prompts import *
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
    prompt_map =
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

def most_similar(id_combinations: dict, similarity_map: dict, num_prompts: int) -> dict(list()):
    prompt_map = defaultdict(list)

    for test_id, train_similarity in similarity_map.items():
        similarity_dict = {t[0]: t[1] for t in train_similarity}
        id_combinations.sort(reverse=True, key=lambda x: sum([similarity_dict[idx] for idx in x]))
        prompt_map[test_id] = id_combinations[:num_prompts]

    return prompt_map

def order_prompts(prompt_map: dict(list()), similarity_map: dict(list()), descending):
    for test_id, train_similarity in similarity_map.items():
        similarity_dict = {t[0]: t[1] for t in train_similarity}
        for train_ids in prompt_map[test_id]:
            train_ids.sort(reverse=descending, key=lambda x: similarity_dict[x])

    return prompt_map


def main():
    parser = argparse.ArgumentParser(description='Generate json dictionary consisting of test_idx: train_indices)')
    parser.add_argument('--dataset', choices=config.tasks)
    parser.add_argument('--experiment_id', type=int)

    parser.add_argument('--in_context', default=2, type=int)
    parser.add_argument('--max_num_prompts', default=1, type=int)
    parser.add_argument('--strategy', choices=['random', 'similar'], default='random', type=str)
    parser.add_argument('--ordering', choices=['random', 'similar-ascending', 'similar-descending'], default='random', type=str)

    parser.add_argument('--uuid', type=str)

    args = parser.parse_args()

    if args.uuid:
        log = get_log_with_uuid(uuid)
        dataset, exp_id, in_context, max_num_prompts, strategy, ordering = log.dataset, log.experiment_id, log.in_context, log.max_num_prompts, log.strategy, log.ordering
    else:
        dataset, exp_id, in_context, max_num_prompts, strategy, ordering = args.dataset, args.experiment_id, args.in_context, args.max_num_prompts, args.strategy, args.ordering
    assert None not in [dataset, exp_id, in_context, max_num_prompts]


    exp_dir = get_dir_with_id(os.path.join(config.experiments,dataset), exp_id)
    gen_id = new_dir_id(exp_dir)

    train_data = read_jsonl(os.path.join(exp_dir, 'train.jsonl'))
    test_data = read_jsonl(os.path.join(exp_dir, 'test.jsonl'))


    prompt_map = {}
    similarity_map = similarity_scores(train_data, test_data, dataset, "all-roberta-large-v1")
    demonstration_ids = [ex["idx"] for ex in train_data]

    if max_num_prompts > 1:
        id_combinations = list([list(x) for x in product(demonstration_ids, repeat=in_context)])

        if strategy == "random":
            id_combinations = random.sample(id_combinations, k=max_num_prompts) if max_num_prompts < len(id_combinations) else id_combinations
            prompt_map = {ex['idx']: id_combinations for ex in test_data}
        else:
            max_num_prompts = min(len(id_combinations), max_num_prompts)
            prompt_map = most_similar(id_combinations, similarity_map, max_num_prompts)

    else: # assume this is the full k-shot baseline
        if strategy == "random":
            demonstration_ids = random.sample(id_combinations, k=in_context)
            prompt_map = {ex['idx']: [demonstration_ids] for ex in test_data}
        else:
            # TODO: clean up redundancy
            for test_id, train_similarity in similarity_map.items():
                similarity_dict = {t[0]: t[1] for t in train_similarity}
                demonstration_ids.sort(reverse=True, key=lambda x: similarity_dict[x])
                prompt_map = demonstration_ids[:in_context]

    if 'similar' in ordering:
        descending = ordering.split("-")[1] == "descending"
        order_prompts(prompt_map, similarity_map, descending)
        max_num_prompts = min(max_num_prompts, max([len(train_ids) for train_ids in prompt_map.values()]))

    generation_dir = os.path.join(exp_dir, config.delim.join([str(x) for x in [gen_id,in_context,max_num_prompts,strategy,*ordering.split("-")]]))
    os.makedirs(generation_dir, exist_ok=True)

    if strategy == 'similar':
        print("Writing similarity scores...", end="")
        write_json(similarity_map, os.path.join(generation_dir, 'similarity_scores.json'))
        print("done!")

    print("Writing prompt map...", end="")
    write_json(prompt_map, os.path.join(generation_dir, 'prompt_map.json'))
    print("done!")

    # log if running with uuid
    if args.uuid:
        log = get_log_with_uuid(args.uuid)
        log.generation_id = str(gen_id)
        log.status = 'inference'
        write_json(log, os.path.join(config.logs, f"{args.uuid}.json"))

if __name__ == '__main__':
    main()
