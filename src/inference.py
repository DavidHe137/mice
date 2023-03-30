#!/usr/bin/env python3
#SBATCH --job-name mice-inference
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/logs/inference/%a.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/logs/inference/%a.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task 6
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --time 15
#SBATCH --requeue

import os
import sys
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append(os.getcwd()) 
from utils import *

# llama imports
from pathlib import Path
from typing import Tuple
import sys
import torch
import time
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

sys.path.append(config.llama)
from llama import ModelArgs, Transformer, Tokenizer, LLaMA

SLURM_ARRAY_TASK_ID = os.getenv('SLURM_ARRAY_TASK_ID')

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def load_llama(
    ckpt_dir: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int
) -> LLaMA:
    tokenizer_path = os.path.join(config.llama, "checkpoints",  "tokenizer.model")

    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def format_prompt(
    test_example, train_examples, dataset
):
    demonstrations = ''.join(list(map(lambda x: format_example(x, dataset, includeLabel=True), train_examples)))
    test_input = format_example(test_example, dataset)

    input_text = demonstrations + test_input

    return input_text

def create_opt_model_input(
    test_example, train_examples, dataset, tokenizer
):
    #TODO: worry about input len at some point
    input_text = format_prompt(test_example, train_examples, dataset)
    input_tokens = tokenizer.encode(input_text, return_tensors="pt").cuda()

    return {"input_text": input_text, "input_tokens": input_tokens}

def create_llama_model_input(
    test_example, train_examples, dataset
):
    #TODO: worry about input len at some point
    demonstrations = ''.join(list(map(lambda x: format_example(x, dataset, includeLabel=True), train_examples)))
    test_input = format_example(test_example, dataset)

    return demonstrations + test_input

def predict_with_llama(
    dataset,
    test_example,
    train_data,
    prompts,
    model,
    max_context_len,
    max_generated_len,
    predictions,  
) -> dict:
    
    # max_input_len = 2048 - max_generated_len

    # make predictions
    prev_num_predictions = len(predictions.keys())
    cur_num_predictions = prev_num_predictions
    for (train_id1, train_id2) in prompts:

        if str((train_id1, train_id2)) in predictions:
            continue

        try:
            input = create_llama_model_input(
                test_example,
                [train_data[train_id1],train_data[train_id2]],
                dataset,
            )
            print(f"Making predictions for prompt={str((train_id1, train_id2))}")
            generated_text = model.generate([input],
                                            max_gen_len=max_generated_len,
                                            temperature=0.8,
                                            top_p=0.95
                                            )

            print("generated_text=", generated_text)

            prediction = extract_prediction(generated_text[0], dataset)
            print("prediction=", prediction)

            predictions[str((train_id1, train_id2))] = {
                "input_text": input,
                "output_text": generated_text,
                "prediction": prediction,
                "label": test_example["label"]
            }
        except Exception as e:
            print(
                f"{str((train_id1, train_id2))} has some problem."
            )
            print(e)
            predictions[str((train_id1, train_id2))] = {
                "input_text": input,
                "output_text": "",
                "prediction": "",
                "label": test_example["label"]
            }
        cur_num_predictions += 1

        # save every 10 examples
        if cur_num_predictions % 10 == 0:
            break

    return (
        predictions,
        cur_num_predictions == prev_num_predictions,
    )

def predict_with_opt(
    dataset,
    test_example,
    train_data,
    prompts,
    model,
    tokenizer,
    max_context_len,
    max_generated_len,
    predictions,
) -> dict:

    # max_input_len = 2048 - max_generated_len

    # make predictions
    prev_num_predictions = len(predictions.keys())
    cur_num_predictions = prev_num_predictions
    for (train_id1, train_id2) in prompts:

        if str((train_id1, train_id2)) in predictions:
            continue

        try:
            input = create_opt_model_input(
                test_example,
                [train_data[train_id1],train_data[train_id2]],
                dataset,
                tokenizer,
            )
            print(f"Making predictions for prompt={str((train_id1, train_id2))}")
            input_len = input["input_tokens"].shape[1]
            outputs = model.generate(
                input["input_tokens"],
                max_new_tokens=max_generated_len,
                temperature=0,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=198,  # special character 'ċ' (bytecode for new line?) NOTE use this for generation
            )
            # generated tokens
            generated_tokens = outputs.sequences[:, input_len:-1]
            # print("generated_tokens=", generated_tokens)

            # generated text
            generated_text = tokenizer.decode(generated_tokens[0])
            print("generated_text=", generated_text)

            prediction = extract_prediction(generated_text.lower(), dataset)
            print("prediction=", prediction)

            predictions[str((train_id1, train_id2))] = {
                "input_text": input["input_text"],
                "output_text": generated_text,
                "prediction": prediction,
                "label": test_example["label"]
            }
        except Exception as e:
            print(
                f"{str((train_id1, train_id2))} has some problem."
            )
            print(e)
            predictions[str((train_id1, train_id2))] = {
                "input_text": input,
                "output_text": "",
                "prediction": "",
                "label": test_example["label"]
            }
        cur_num_predictions += 1

        # save every 10 examples
        if cur_num_predictions % 10 == 0:
            break

    return (
        predictions,
        cur_num_predictions == prev_num_predictions,
    )

def predict_batch(
    tokenizer,
    model,
    test_example:dict,
    prompts:list(list(str)),
    train_data,
    dataset:str,
    max_generated_len:int,
    batch_size: int,
):    

    train_ids = [tuple(ids) for ids in prompts] 
    preds = []

    input_text = [format_prompt(test_example, 
                                [train_data[train_id] for train_id in prompt], 
                                dataset, tokenizer) for prompt in prompts]
    batch_tokens = tokenizer(input_text, padding=True, return_tensors="pt").to('cuda:0')

    num_batches = round(batch_tokens['input_ids'].shape[0] / batch_size + 0.5)
    for batch in batch_tokens['input_ids'].chunk(num_batches):
        outputs = model.generate(
            batch,
            max_new_tokens=max_generated_len,
            temperature=0,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=198,  # special character 'ċ' (bytecode for new line?) NOTE use this for generation
        )

        preds.extend(tokenizer.batch_decode(outputs.sequences[:, -max_generated_len:]))
    
    results = {x[0]: x[1] for x in zip(train_ids, preds)}

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_id', type=str)
    parser.add_argument('experiment_id', type=str)
    parser.add_argument('generation_id', type=str)

    parser.add_argument('model', type=str.lower)

    parser.add_argument('--uuid', type=str)

    args = parser.parse_args()

    experiment_id = args.experiment_id
    generation_id = args.generation_id
    if args.uuid:
        log = get_log_with_uuid(args.uuid)
        experiment_id = log['experiment_id']
        generation_id = log['generation_id']

    exp_info = get_experiment_info(experiment_id)



    test_ids = [exp_info['test_ids'][i] for i in range(int(SLURM_ARRAY_TASK_ID), int(SLURM_ARRAY_TASK_ID) + config.tests_per_gpu)]
    print(test_ids)

    train_data = read_jsonl(os.path.join(exp_info['location'], 'train.jsonl'))
    test_data = read_jsonl(os.path.join(exp_info['location'], 'test.jsonl'))

    train_data = {ex["idx"]: ex for ex in train_data}
    test_data = {ex["idx"]: ex for ex in test_data}

    test_examples = [test_data[i] for i in test_ids]
    print(test_examples)




    models = {"opt-125m": "facebook/opt-125m",
              "opt-350m": "facebook/opt-350m",
              "opt-1.3b": "facebook/opt-1.3B",
              "opt-2.7b": "facebook/opt-2.7B",
              "opt-6.7b": "facebook/opt-6.7B",
              "llama-7b": "7B"}
    
    model_name = models[args.model]
    max_generated_len = 16 #NOTE: for SuperGLUE
    max_context_len = 2048

    batch_sizes = {"opt-125m": 64,
              "opt-350m": 32,
              "opt-1.3b": 16,
              "opt-2.7b": 8,
              "opt-6.7b": 4,
              "llama-7b": 4}
    batch_size = batch_sizes[args.model]

    print(f"Load {args.model}...", end="")
    model = None
    if "llama" in args.model:
        #TODO: optimize llama inference
        max_seq_len = 1024
        max_batch_size = 32

        ckpt_dir = os.path.join(config.llama, "checkpoints", model_name)

        local_rank, world_size = setup_model_parallel()
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        model = load_llama(
            ckpt_dir, local_rank, world_size, max_seq_len, max_batch_size
        )
    else:    
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    print("done!")


    generation_dir = exp_info['generations'][generation_id]['location']
    example_dir = os.path.join(generation_dir, args.model, test_id)
    os.makedirs(example_dir, exist_ok=True)

    predictions = {}
    predictions_filepath = os.path.join(example_dir, "predictions.json")
    if os.path.exists(predictions_filepath):
        print("Recover predictions...", end="")
        with open(predictions_filepath, "r") as f:
            predictions = json.load(f)
        print("done!")

    print("Get prompt_map...", end="")
    prompt_map_filepath = os.path.join(generation_dir, "prompt_map.json")
    with open(prompt_map_filepath, "r") as f:
        prompt_map = json.load(f)
    print("done!")

    # make predictions
    end_prediction = False
    while not end_prediction:
        if "llama" in args.model:
            predictions, end_prediction = predict_with_llama(
            exp_info['dataset'],
            test_example,
            train_data,
            prompt_map[test_id],
            model,
            max_context_len,
            max_generated_len,
            predictions,
            )
        else:
            predictions, end_prediction = predict_with_opt(
                exp_info['dataset'],
                test_example,
                train_data,
                prompt_map[test_id],
                model,
                tokenizer,
                max_context_len,
                max_generated_len,
                predictions,
            )

        print("Save %s predictions..." % len(predictions.keys()), end="")

        # save outputs and monitor information
        with open(predictions_filepath, "w") as f:
            json.dump(predictions, f, indent=4)
        monitor_filepath = os.path.join(example_dir, "monitoring.json")
        with open(monitor_filepath, "w") as f:
            json.dump({"num_predictions": len(predictions.keys())}, f, indent=4)
        print("done!")


if __name__ == "__main__":
    main()
