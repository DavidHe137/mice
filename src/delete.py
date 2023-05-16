import os
import argparse
import random
from datetime import datetime
from copy import deepcopy

from utils import *
import config

def main():      
    '''
    Delete an experiment, or a generation within an experiment id
    '''
    parser = argparse.ArgumentParser(description='Configure dataset.')

    parser.add_argument('experiment_id', type=str)
    parser.add_argument('--generation_id', type=str)

    args = parser.parse_args()
    exp_info = get_experiment_info(args.experiment_id)

    uuids = []
    # only delete for generation
    if args.generation_id:
        for file in os.listdir(config.logs):
            if os.path.isfile(file) and file.endswith(".json"):
                log = read_json(file)
                if (args.generation_id
                    and log['generation_id'] == args.generation_id 
                    and log['experiment_id'] == args.experiment_id):
                    os.remove(file)

    # delete entire experiment
    else:
        for file in os.listdir(config.logs):
            if os.path.isfile(file) and file.endswith(".json"):
                log = read_json(file)
                if log['experiment_id'] == args.experiment_id:
                    os.remove(file)
                    uuids.append(os.path.basename(file))


    #delete logs


    #delete from summary & info

        generation_dir = os.path.join(exp_info['location'], 'generations', f"{args.ordering}_{args.in_context}_{args.max_num_prompts}_{args.encoder}")


    #delete folders

    

        
if __name__ == '__main__':
    main()
