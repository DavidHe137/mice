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

sys.path.append("/coc/pskynet6/dhe83/mice/src")
from utils import *
import config
from prompts import *

def main():
    '''
    Create experiment folder, set up train/test splits, log details
    '''
    parser = argparse.ArgumentParser(description='Configure dataset.')

    parser.add_argument('--dataset', choices=config.tasks)
    parser.add_argument('--train', type=int)
    parser.add_argument('--test', type=int, help="Pass 0 for all test examples")
    parser.add_argument('--uuid', type=str)

    args = parser.parse_args()

    if args.uuid:
        log = get_log_with_uuid(args.uuid)
        dataset, train, test = log.dataset, log.train, log.test
    else:
        dataset, train, test = args.dataset, args.train, args.test

    assert None not in [dataset, train, test]


    exp_dir = os.path.join(config.experiments, dataset)
    os.makedirs(exp_dir, exist_ok=True)

    exp_id = new_dir_id(exp_dir)
    # read data
    data_dir = os.path.join(config.data, args.dataset)
    train_data = read_jsonl(os.path.join(data_dir, 'train.jsonl'))
    test_data = read_jsonl(os.path.join(data_dir, 'val.jsonl'))

    # sampling bounds
    train = min(len(train_data), train)
    test = min(len(test_data), test) if test > 0 else len(test_data)

    # sample k demonstrations for n test examples
    train_data = random.sample(train_data, k=train)
    test_data = random.sample(test_data, k=test)

    # sort by idx
    train_data.sort(key=lambda x: x['idx'])
    test_data.sort(key=lambda x: x['idx'])

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

    # log if running with uuid
    if args.uuid:
        log = get_log_with_uuid(args.uuid)
        log.experiment_id = str(exp_id)
        log.status = 'prompt_generation'
        write_json(log, os.path.join(config.logs, f"{args.uuid}.json"))

if __name__ == '__main__':
    main()
