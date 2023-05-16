#!/usr/bin/env python3
#SBATCH --job-name mice-setup
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/logs/setup/%A.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/logs/setup/%A.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --time 5
#SBATCH --requeue

import os
import sys
import argparse
import random
from datetime import datetime
from copy import deepcopy

sys.path.append("/coc/pskynet6/dhe83/mice/src")
from utils import *
import config
from prompts import *

def preprocess_MultiRC(examples:list):
    for ex in examples:
        ex = ex['passage']
        for q in ex['questions']:
            q["answers"] = {a["idx"]: a for a in q["answers"]}
        ex["questions"]= {q["idx"]: q for q in ex["questions"]}

def preprocess_ReCoRD(examples:list):
    for ex in examples:
        ex["qas"]= {q["idx"]: q for q in ex["qas"]}

def main():
    '''
    Create experiment folder, set up train/test splits, log details
    '''
    parser = argparse.ArgumentParser(description='Configure dataset.')

    parser.add_argument('--dataset', choices=config.tasks)
    parser.add_argument('--train', type=int)
    parser.add_argument('--test', type=int, help="Pass 0 for all test examples")
#   parser.add_argument('--uuid', type=str)

    args = parser.parse_args()

    dataset, train, test = args.dataset, args.train, args.test
    assert None not in [dataset, train, test]


    exp_dir = os.path.join(config.experiments, dataset)
    os.makedirs(exp_dir, exist_ok=True)

    exp_id = new_dir_id(exp_dir)
    # read data
    data_dir = os.path.join(config.data, args.dataset)
    train_data = read_jsonl(os.path.join(data_dir, 'train.jsonl'))
    test_data = read_jsonl(os.path.join(data_dir, 'val.jsonl')) # NOTE: original datasets only have labels for validation

    # Winograd preprocessing
    if dataset == "Winograd":
        train_data = [ex for ex in train_data if ex['label']]
        test_data = [ex for ex in test_data if ex['label']]

    # sampling bounds
    train = min(len(train_data), train)
    test = min(len(test_data), test) if test > 0 else len(test_data)

    # sample k demonstrations for n test examples
    train_data = random.sample(train_data, k=train)
    test_data = random.sample(test_data, k=test)

    # sort by idx
    train_data.sort(key=lambda x: x['idx'])
    test_data.sort(key=lambda x: x['idx'])

    # MultiRC, ReCoRD preprocessing
    if dataset == 'MultiRC':
        preprocess_MultiRC(train_data)
        preprocess_MultiRC(test_data)
    elif dataset == 'ReCoRD':
        preprocess_ReCoRD(train_data)
        preprocess_ReCoRD(test_data)

    # format in-context examples
    for ex in train_data:
        ex['in_context'] = format_in_context(ex, dataset)

    exp_dir = os.path.join(exp_dir, config.delim.join([str(x) for x in [exp_id, train, test]]))

    summary = {
        'created': str(datetime.now()),
        'dataset': args.dataset,
        'train': args.train,
        'test': args.test,
        'train_ids': [example['idx'] for example in train_data],
        'test_ids': [example['idx'] for example in test_data]
    }

    # write directories/files
    os.makedirs(exp_dir, exist_ok=True)
    write_jsonl(train_data, os.path.join(exp_dir, 'train.jsonl'))
    write_jsonl(test_data, os.path.join(exp_dir, 'test.jsonl'))
    write_json(summary, os.path.join(exp_dir, 'summary.json'))

#   # log if running with uuid
#   if args.uuid:
#       summary['uuid'] = args.uuid
#       summary['experiment_id'] = str(exp_id)
#       del summary['runs']
#       summary['status'] = 'prompt_generation'
#       write_json(summary,os.path.join(config.logs, f"{args.uuid}.json"))

if __name__ == '__main__':
    main()
