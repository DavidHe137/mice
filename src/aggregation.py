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
from prompts import *
import config

'''
    Preprocessing
'''
def check_missed(missed_predictions, example_predictions):
        # Count & delete missed prompts
        missed = 0
        for prompt, pred in example_predictions.items():
            if pred['prediction'] == "":
                missed+=1
                del example_predictions[prompt]
        if missed > 0:
            missed_predictions[example_id] = missed

'''
    Mixture Weights
'''

def similarity_weights(
    predictions: dict, prompt_map: list, similarity_lst: list, temperature
) -> dict:
    '''
    For an implied test example, calculates normalized similarity_score weighting for each prompt
    '''
    # first convert similarity_lst to dict mapping id to score
    similarity_dict = dict(similarity_lst)


    all_prompt_scores = dict()
    for prompt_id in prompt_map:
        key = "|".join([str(x) for x in prompt_id])
        assert(key) in predictions

        all_prompt_scores[key] = sum(
           [similarity_dict[train_id] for train_id in prompt_id]
        ) / len(prompt_id)


    # then get the list of prompt scores. Technically this step is unnecessary,
    # but put here for sanity check
    prompt_ids = list(predictions.keys())
    prompt_scores = [all_prompt_scores[prompt_id] for prompt_id in prompt_ids]

    # softmax temperature
    prompt_scores = torch.tensor(prompt_scores) / temperature
    prompt_probs = prompt_scores.softmax(dim=-1)

    # map it back to a dictionary
    prompt_probs = {
        prompt_id: prompt_probs[i].item() for i, prompt_id in enumerate(prompt_ids)
    }

    return prompt_probs

def compute_and_save_weights(
    example_dir, example_predictions, prompt_map, similarity_lst, temperature
):
    # check if file already exists
    weights_filepath = os.path.join(example_dir, f"similarity_weights_{temperature}.json")
    if os.path.exists(weights_filepath):
        weights = read_json(weights_filepath)
    else:
        weights = similarity_weights(
            example_predictions, prompt_map, similarity_lst, temperature
        )
        write_json(weights, weights_filepath)

    return weights

'''
    Count-based (Sampling)
'''

def ensemble_counts(predictions: dict, gold_label: str) -> dict:
    '''
    Counts the number of prompts in an ensemble that generates an answer span
    '''
    # all_counts = {"span": list(prompt_ids)}
    all_counts = defaultdict(list)
    for prompt_id, v in predictions.items():
        # TODO: list of answer spans
        all_counts[v['prediction']].append(prompt_id)

    # sort by number of prompts, descending
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

def sampling(prompt_probs, sampled_probs, gold_label):
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

def ensemble_confidence(predictions:dict, gold_label: str) -> dict:
    '''
    Maps dict {prompt_id: {label: probability}}
    '''

    token_probs = {prompt_id: v['probs'] for prompt_id, v in predictions.items()}
    all_probs = {}

    for probs in token_probs.values():
        for label, prob in probs.items():
            if label not in all_probs:
                all_probs[label] = 0
            all_probs[label] += prob

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

    return new_all_probs_lst, token_probs

def dict_softmax(d:dict(), temperature: float):
    mapping = [(k, v) for k, v in d.items()]
    vals = torch.tensor([x[1] for x in mapping], dtype=torch.float32) / temperature
    normalized = torch.log_softmax(vals, dim=-1)
    mapping = {e[0]: normalized[i].item() for i, e in enumerate(mapping)}
    return mapping

def confidence(prompt_probs, token_probs, gold_label, temperature):
    # Compute P(y|Z) from all the Indicator(m \in Z_i) and P(Z_i), for all i
    # NOTE: This is the crux computation-- all others are flowery
    all_probs = defaultdict(float)
    for prompt_id, label_probs in token_probs.items():
        for label, prob in label_probs.items():
            if label not in all_probs:
                all_probs[label] = 0
            all_probs[label] += prompt_probs[prompt_id] * prob

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
    parser.add_argument('--dataset', choices=config.tasks)
    parser.add_argument('--experiment_id', type=int)
    parser.add_argument('--generation_id', type=int)
    parser.add_argument('--model', type=str.lower)
    parser.add_argument('--method', default="mice-sampling", choices=['mice-sampling', 'sampling', 'mice-confidence', 'confidence'], type=str)
    parser.add_argument('--temperature', default=1.0, type=float)
#   parser.add_argument('--uuid', type=str)

    args = parser.parse_args()
    dataset, exp_id, gen_id, model, method, temperature = args.dataset, args.experiment_id, args.generation_id, args.model, args.method, args.temperature

#   if args.uuid:
#       log_file = os.path.join(config.logs, f"{args.uuid}.json")
#       log = read_json(log_file)
#       log['last_modified'] = str(datetime.now())
#       log['status'] = "aggregation"
#       write_json(log, log_file)

#       experiment_id = log['experiment_id']
#       generation_id = log['generation_id']

    exp_dir = get_dir_with_id(os.path.join(config.experiments, dataset), exp_id)
    test_data = read_jsonl(os.path.join(exp_dir, 'test.jsonl'))

    test_data = {ex["idx"]: ex for ex in test_data}

    generation_dir = get_dir_with_id(exp_dir, gen_id)

    examples_dir = os.path.join(generation_dir, args.model)

    if "mice" in method:
        similarity_map = read_json(os.path.join(generation_dir, "similarity_scores.json"))

    prompt_map = read_json(os.path.join(generation_dir, "prompt_map.json"))

    # generate mention_counts
    predictions = dict()
    failed_predictions = []
    missed_predictions = {}
    for example_id, example in test_data.items():

        example_dir = os.path.join(examples_dir, str(example_id))
        predictions_filepath = os.path.join(example_dir, "predictions.json")

        # Preprocessing
        if not os.path.exists(predictions_filepath):
            print("example_id=%s does not have predictions" % example_id)
            failed_predictions.append(example_id)
            continue

        example_predictions = read_json(predictions_filepath)
        check_missed(missed_predictions, example_predictions)

        gold_label = test_data[example_id]['label']

        # Ensembling
        pred = {}
        if 'sampling' in method:
            model_probs = ensemble_counts(example_predictions, gold_label)
            predictions[example_id] = {
                "input_text": format_example(example, dataset),
                "prediction": model_probs[0]['span'] if len(model_probs) > 0 else "",
                "num_prompts": model_probs[0]['num_prompts'] if len(model_probs) > 0 else "",
                "label": gold_label,
            }


        else:
            model_probs, token_probs = ensemble_confidence(example_predictions, gold_label)
            predictions[example_id] = {
                "input_text": format_example(example, dataset),
                "prediction": model_probs[0]['span'] if len(model_probs) > 0 else "",
                "num_prompts": model_probs[0]['prob'] if len(model_probs) > 0 else "",
                "label": gold_label,
            }

        write_json(  # \mathbb{1}(m \ in Y_{x,z})
                model_probs,
                os.path.join(example_dir, f"{args.method}_counts.json")
        )

        if 'mice' not in method:
            continue

        # Weighted Ensembling
        prompt_probs = compute_and_save_weights(
            example_dir,
            example_predictions,
            prompt_map[str(example_id)],
            similarity_map[str(example_id)],
            temperature
        )

        if 'sampling' in method:
            combined_probs = sampling(prompt_probs, model_probs, gold_label)
        else:
            combined_probs = confidence(prompt_probs, token_probs, gold_label, temperature)

        predictions[example_id] = {
            "input_text": format_example(example, dataset),
            "prediction": combined_probs[0]['span'] if len(combined_probs) > 0 else "",
            "label": gold_label,
        }

        write_json(  # P(y_i|z,x)
            combined_probs,
            os.path.join(example_dir, f"{args.method}|{temperature}|weighted.json")
        )

#   print("predictions", [p['prediction'] for p in predictions.values()])
#   print("references", [p['label'] for p in predictions.values()])

#   for k, v in predictions.items():
    #   if validate(v['prediction'], v['label'], dataset):
#       print(v['prediction'], "|",  v['label'])

    correct = 0
    total = 0
    for p in predictions.values():
        if validate(p['prediction'],p['label'], dataset):
            correct+=1
        total+=1

    result = float(correct) / total
    print(result)

    predictions = {"accuracy": result, "predictions": predictions}
    predictions_filepath = os.path.join(generation_dir, args.model, f"{args.method}_predictions.json")
    write_json(predictions, predictions_filepath)

#   run_id = len(exp_info['runs']) + 1
#   if args.uuid:
#       run_id = args.uuid

#   run = {
#       'evaluated': str(datetime.now()),
#       'generation': exp_info['generations'][generation_id],
#       'model': args.model,
#       'method': args.method,
#       'result': result,
#       'failed_predictions': failed_predictions
#   }

#   exp_summary = os.path.join(config.experiments, 'summary.json')
#   exp_summary_data = read_json(exp_summary)
#   exp_summary_data[experiment_id]['runs'][run_id] = run
#   write_json(exp_summary_data, exp_summary)

#   exp_info['runs'][run_id] = run
#   write_json(exp_info, os.path.join(exp_summary_data[experiment_id]['location'], 'info.json'))

#   if args.uuid:
#       log_file = os.path.join(config.logs, f"{args.uuid}.json")
#       log = read_json(log_file)
#       log['last_modified'] = str(datetime.now())
#       log['status'] = "finished"
#       write_json(log, log_file)

if __name__ == "__main__":
    main()
