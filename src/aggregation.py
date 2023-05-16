#!/usr/bin/env python3
#SBATCH --job-name mice-aggregation
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/logs/aggregation/%A.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/logs/aggregation/%A.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --time 5
#SBATCH --requeue

'''
Aggregates prompts. Evaluates as well.
'''
from collections import defaultdict
import os
import sys
import argparse
from datetime import datetime

import torch

sys.path.append("/coc/pskynet6/dhe83/mice/src") 
from utils import *
import config

def produce_mention_probs(predictions: dict, gold_label: str, dataset: str) -> dict:
    # first produce all indicators
    all_counts = defaultdict(list)
    for prompt_id, v in predictions.items():
        all_counts[verbalize(v['prediction'], dataset)].append(prompt_id)

    # convert to list and sorted
    all_counts_lst = [(k, v) for k, v in all_counts.items()]
    all_counts_lst = sorted(all_counts_lst, key=lambda x: len(x[1]), reverse=True)

    # convert to dict
    new_all_counts_lst = []
    for (k, v) in all_counts_lst:
        new_all_counts_lst.append(
            {
                "span": k,
                "prompts": v,
                "num_prompts": len(v),
                "is_gold": k is gold_label,
            }
        )

    return new_all_counts_lst

def compute_prompt_probs_similar(
    predictions: dict, prompt_map: list, similarity_lst: list
) -> dict:

    # first convert similarity_lst to dict mapping id to score
    similarity_dict = dict(similarity_lst)

    # first compute \sum_i s_i. Normalized by length of prompt_id
    all_prompt_scores = dict()
    for prompt_id in prompt_map:
        prompt_str = str(tuple(prompt_id))
        assert(prompt_str) in predictions

        all_prompt_scores[prompt_str] = sum(
            [similarity_dict[train_id] for train_id in prompt_id]
        ) / len(prompt_id)

    # then get the list of prompt scores. Technically this step is unnecessary,
    # but put here for sanity check
    prompt_ids = list(predictions.keys())
    prompt_scores = [all_prompt_scores[prompt_id] for prompt_id in prompt_ids]

    # softmax to get the probabilities

    prompt_probs = torch.tensor(prompt_scores).softmax(dim=-1)

    # map it back to a dictionary
    prompt_probs = {
        prompt_id: prompt_probs[i].item() for i, prompt_id in enumerate(prompt_ids)
    }

    return prompt_probs

def compute_and_save_priors(
    example_dir, example_predictions, prompt_map, similarity_lst
):
    # check if file already exists
    priors_filepath = os.path.join(example_dir, "similar_priors.json")
    if os.path.exists(priors_filepath):
        priors = read_json(priors_filepath)
    else:
        priors = compute_prompt_probs_similar(
            example_predictions, prompt_map, similarity_lst
        )
        write_json(priors, priors_filepath)

    return priors

def sampling(sampled_probs, prompt_probs, gold_label):
    # Compute P(y|Z) from all the Indicator(m \in Z_i) and P(Z_i), for all i
    # NOTE: This is the crux computation-- all others are flowery
    all_probs = defaultdict(float)
    for mention_d in sampled_probs:
        all_probs[mention_d["span"]] = sum(
            [prompt_probs[prompt] for prompt in mention_d["prompts"]]
        )

    # convert to list and sorted
    all_probs_lst = [(k, v) for k, v in all_probs.items()]
    all_probs_lst = sorted(all_probs_lst, key=lambda x: x[1], reverse=True)

    # convert to dict
    new_all_probs_lst = []
    for (k, v) in all_probs_lst:
        new_all_probs_lst.append(
            {
                "span": k,
                "prob": v,
                "is_gold": k is gold_label,
            }
        )

    return new_all_probs_lst

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment_id', type=str)
    parser.add_argument('generation_id', type=str)
    parser.add_argument('model', type=str.lower)
    parser.add_argument('--method', default="mice-sampling", choices=['mice-sampling', 'majority-vote'], type=str)
    parser.add_argument('--uuid', type=str)

    args = parser.parse_args()

    experiment_id = args.experiment_id
    generation_id = args.generation_id
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

    # generate mention_counts
    predictions = dict()
    failed_predictions = []
    missing_predictions = {}
    for example_id, example in test_data.items():

        example_dir = os.path.join(examples_dir, str(example_id))
        predictions_filepath = os.path.join(example_dir, "predictions.json")

        # read in predictions
        if not os.path.exists(predictions_filepath):
            print("example_id=%s does not have predictions" % example_id)
            failed_predictions.append(example_id)
            continue
        example_predictions = read_json(predictions_filepath)

        # Count & delete missing prompts
        missing = 0
        for prompt, pred in example_predictions.items():
            if pred['prediction'] == "":
                missing+=1
                del example_predictions[prompt]
        if missing > 0:
            missing_predictions[example_id] = missing

        # produce raw mention probs (\mathbb{1}(m \in Y_{z,x}))
        gold_label = test_data[example_id]['label']
        sampled_probs = produce_mention_probs(example_predictions, gold_label, exp_info['dataset'])

        if args.method == "majority-vote": 
            predictions[example_id] = {
                "input_text": format_example(example, exp_info['dataset']),
                "prediction": sampled_probs[0]['span'] if len(sampled_probs) > 0 else "",
                "label": gold_label,
            }
            continue

        # outputs the counts and predictions
        write_json(  # \mathbb{1}(m \ in Y_{x,z})
            sampled_probs,
            os.path.join(example_dir, f"{args.method}_sampled_probs.json")
        )

        # compute P(Z_i|X) and save it
        prompt_probs = compute_and_save_priors(
            example_dir,
            example_predictions,
            prompt_map[str(example_id)],
            similarity_map[str(example_id)],
        )

        # produce raw mention info
        combined_probs = sampling(
            sampled_probs, prompt_probs, gold_label
        )

        predictions[example_id] = {
            "input_text": format_example(example, exp_info['dataset']),
            "prediction": combined_probs[0]['span'] if len(combined_probs) > 0 else "",
            "label": gold_label,
        }

        write_json(  # P(y_i|z,x)
            combined_probs,
            os.path.join(example_dir, f"{args.method}_combined_probs.json")
        )

    # output the predictions
    predictions_filepath = os.path.join(generation_dir, args.model, f"{args.method}_predictions.json")
    write_json(predictions, predictions_filepath)

    print(args.method)
    print("predictions", [p['prediction'] for p in predictions.values()])

    if exp_info['dataset'] == 'RTE':
        for k in predictions.keys():
            predictions[k]['label'] = verbalize(predictions[k]['label'], exp_info['dataset'])
            
    print("references", [p['label'] for p in predictions.values()])

    correct = 0
    total = 0
    for p in predictions.values():
        if p['prediction'] == p['label']:
            correct+=1
        total+=1

    result = float(correct) / total
    
    run_id = len(exp_info['runs']) + 1
    if args.uuid:
        run_id = args.uuid

    run = {
        'evaluated': str(datetime.now()),
        'generation': exp_info['generations'][generation_id],
        'model': args.model,
        'method': args.method,
        'result': result,
        'failed_predictions': failed_predictions
    }

    exp_summary = os.path.join(config.experiments, 'summary.json')
    exp_summary_data = read_json(exp_summary)
    exp_summary_data[experiment_id]['runs'][run_id] = run
    write_json(exp_summary_data, exp_summary)

    exp_info['runs'][run_id] = run
    write_json(exp_info, os.path.join(exp_summary_data[experiment_id]['location'], 'info.json'))

    if args.uuid:
        log_file = os.path.join(config.logs, f"{args.uuid}.json")
        log = read_json(log_file)
        log['last_modified'] = str(datetime.now())
        log['status'] = "finished"
        write_json(log, log_file)
    
if __name__ == "__main__":
    main()