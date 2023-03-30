'''
Aggregates prompts. Evaluates as well.
'''
from collections import defaultdict
import os
import argparse
from datetime import datetime

from utils import *
import config

def demonstration_performance():
    print("hi")

def pareto_curves():
    print("hi")

def probability_of_gold():

def probability_of_gold_after_similarity():




def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('uuid', type=str)

    args = parser.parse_args()

    

    for 
    if args.uuid:
        log_file = os.path.join(config.logs, f"{args.uuid}.json")
        log = read_json(log_file)
        log['last_modified'] = str(datetime.now())
        log['status'] = "aggregation"
        write_json(log, log_file)

        experiment_id = log['experiment_id']
        generation_id = log['generation_id']

    exp_info = get_experiment_info(experiment_id)
    generation_dir = exp_info['generations'][generation_id]['location']        

    examples_dir = os.path.join(generation_dir, args.model)
    similarity_map = read_json(os.path.join(generation_dir, "similarity_scores.json"))
    prompt_map = read_json(os.path.join(generation_dir, "prompt_map.json"))

    test_data = read_jsonl(os.path.join(exp_info['location'], 'test.jsonl'))  
    test_data = {ex["idx"]: ex for ex in test_data}  

#which in
if __name__ == "__main__":
    main()