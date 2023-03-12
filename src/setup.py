import os
import argparse
import random
from datetime import datetime
from copy import deepcopy

from utils import *
import config

def main():      
    '''
    Create experiment folder, set up train/test splits, log details
    '''
    parser = argparse.ArgumentParser(description='Configure dataset.')

    parser.add_argument('--dataset', choices=['BoolQ', 'COPA', 'RTE', 'WiC', 'WSC'])
    parser.add_argument('--train', type=int)
    parser.add_argument('--test', type=int, help="Pass 0 for all test examples")
    parser.add_argument('--uuid', type=str)

    args = parser.parse_args()

    # gather details
    exp_summary = os.path.join(config.experiments, 'summary.json') 
    exp_id = 1

    exp_summary_data = {}
    if os.path.exists(exp_summary):        
        exp_summary_data = read_json(exp_summary)
        exp_id = max([int(_) for _ in exp_summary_data.keys()]) + 1  

    # read data      
    data_dir = os.path.join(config.data, args.dataset)
    train_data = read_jsonl(os.path.join(data_dir, 'train.jsonl'))
    test_data = read_jsonl(os.path.join(data_dir, 'val.jsonl')) # NOTE: original datasets only have labels for validation

    # sample k demonstrations for n test examples
    # TODO: bounds for sampling numbers
    train_data = random.choices(train_data, k=args.train)
    if args.test > 0: # pass < 1 to test all examples
        test_data = random.choices(test_data, k=args.test)
    else:
        args.test = len(test_data)
    train_data.sort(key=lambda x: x['idx'])
    test_data.sort(key=lambda x: x['idx'])

    exp_dir = os.path.join(config.experiments, args.dataset,
                    f"id_{exp_id}_train_{args.train}_test_{args.test}")

    summary = {
        'created': str(datetime.now()),
        'location': exp_dir,
        'dataset': args.dataset,
        'train': args.train,
        'test': args.test,
        'runs': {}
    }

    info = deepcopy(summary)
    info['id'] = str(exp_id)
    info['generations'] = {}
    info['train_ids'] = [example['idx'] for example in train_data]
    info['test_ids'] = [example['idx'] for example in test_data]

    # write directories/files
    os.makedirs(exp_dir, exist_ok=True)
    write_jsonl(train_data, os.path.join(exp_dir, 'train.jsonl'))
    write_jsonl(test_data, os.path.join(exp_dir, 'test.jsonl'))
    write_txt([str(example['idx']) for example in test_data], os.path.join(exp_dir, 'test_ids.txt'))

    exp_summary_data[exp_id] = summary
    write_json(exp_summary_data, exp_summary)
    write_json(info, os.path.join(exp_dir, 'info.json'))

    # log if running with uuid
    if args.uuid:
        summary['uuid'] = args.uuid
        summary['experiment_id'] = str(exp_id)
        del summary['runs']
        summary['status'] = 'prompt_generation'
        write_json(summary,os.path.join(config.logs, f"{args.uuid}.json"))
        
if __name__ == '__main__':
    main()
