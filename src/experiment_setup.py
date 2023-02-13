import os
from pathlib import Path
import argparse
from utils import *

import random
from datetime import datetime
from copy import deepcopy
from shutil import copyfile

def main():      
    '''
    Create experiment folder, set up train/test splits, log details
    '''
    parser = argparse.ArgumentParser(description='Configure dataset, model-size, method, data splits.')
    parser.add_argument('--model_size', choices=['350m'])
    parser.add_argument('--method', choices=['mice_sampling'])
    parser.add_argument('--in_context', default='1', type=int)
    parser.add_argument('--max_num_prompts', type=int)

    parser.add_argument('--dataset', choices=['BoolQ'])
    parser.add_argument('--train', type=int)
    parser.add_argument('--test', type=int)

    parser.add_argument('--reference_experiment_id', default=0, type=int) # only positive experiment IDs


    args = parser.parse_args()

    # gather absolute paths
    project_root = Path(__file__).resolve().parents[1]
    data_home = os.path.join(project_root, 'data')
    exp_home = os.path.join(project_root, 'experiments')

    # gather details
    exp_summary = os.path.join(exp_home, 'summary.json') 
    exp_id = 1

    exp_summary_data = {'summary': []}
    if os.path.exists(exp_summary):        
        exp_summary_data = read_json(exp_summary)
        past_exp_ids = {exp['id'] for exp in exp_summary_data['summary']}
        exp_id = max(past_exp_ids) + 1        

    # choose same examples as --reference_experiment
    if args.reference_experiment_id > 0:
        # find reference experiment data
        exp_summary_data = read_json(exp_summary)
        ref_exp = list(filter(lambda x: x['id'] == args.reference_experiment_id, exp_summary_data['summary']))
        assert len(ref_exp) == 1
        ref_exp = ref_exp[0]

        exp_dir = os.path.join(exp_home, args.dataset, args.model_size, args.method, 
                f"train{ref_exp['train']}_test{ref_exp['test']}_id{exp_id}") 

        summary = {
            'id': exp_id,
            'created': str(datetime.now()),
            'location': exp_dir,
            'dataset': ref_exp['dataset'],
            'model_size': args.model_size,
            'method': args.method,
            'in_context': args.in_context,
            'max_num_prompts': args.max_num_prompts,
            'train': ref_exp['train'],
            'test': ref_exp['test']
        }
        info = deepcopy(summary)

        ref_exp_dir = ref_exp['location']
        ref_exp_info = read_json(os.path.join(ref_exp_dir, 'info.json'))
        info['train_ids'] = ref_exp_info['train_ids']
        info['test_ids'] = ref_exp_info['test_ids']

        os.makedirs(exp_dir, exist_ok=True)
        copyfile(os.path.join(ref_exp_dir, 'train.jsonl'), os.path.join(exp_dir, 'train.jsonl'))
        copyfile(os.path.join(ref_exp_dir, 'test.jsonl'), os.path.join(exp_dir, 'test.jsonl'))
        copyfile(os.path.join(ref_exp_dir, 'test_ids.txt'), os.path.join(exp_dir, 'test_ids.txt'))

    else:
        data_dir = os.path.join(data_home, args.dataset)
        train_data = read_jsonl(os.path.join(data_dir, 'train.jsonl'))
        test_data = read_jsonl(os.path.join(data_dir, 'val.jsonl')) # original datasets only have labels for validation

        # sample k demonstrations for n test examples
        train_data = random.choices(train_data, k=args.train)
        if args.test > 0: # pass -1 to test all examples
            test_data = random.choices(test_data, k=args.test)
        else:
            args.test = len(test_data)
        train_data.sort(key=lambda x: x['idx'])
        test_data.sort(key=lambda x: x['idx'])

        exp_dir = os.path.join(exp_home, args.dataset, args.model_size, args.method, 
                        f"train{args.train}_test{args.test}_id{exp_id}")

        summary = {
            'id': exp_id,
            'created': str(datetime.now()),
            'location': exp_dir,
            'dataset': args.dataset,
            'model_size': args.model_size,
            'method': args.method,
            'in_context': args.in_context,
            'max_num_prompts': args.max_num_prompts,
            'train': args.train,
            'test': args.test
        }
        info = deepcopy(summary)

        info['train_ids'] = [example['idx'] for example in train_data]
        info['test_ids'] = [example['idx'] for example in test_data]

        os.makedirs(exp_dir, exist_ok=True)
        write_jsonl(train_data, os.path.join(exp_dir, 'train.jsonl'))
        write_jsonl(test_data, os.path.join(exp_dir, 'test.jsonl'))
        write_txt([str(example['idx']) for example in test_data], os.path.join(exp_dir, 'test_ids.txt'))

    exp_summary_data['summary'].append(summary)
    write_json(exp_summary_data, exp_summary)
    write_json(info, os.path.join(exp_dir, 'info.json'))
        
if __name__ == '__main__':
    main()