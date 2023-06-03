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
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
from math import ceil

sys.path.append("/coc/pskynet6/dhe83/mice/src")
from utils import *
from prompts import *

# llama imports
from pathlib import Path
from typing import Tuple
import sys
import torch
import time
# from fairscale.nn.model_parallel.initialize import initialize_model_parallel

#sys.path.append(config.llama)
# from llama import ModelArgs, Transformer, Tokenizer, LLaMA

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

#   def load_llama(
#       ckpt_dir: str,
#       local_rank: int,
#       world_size: int,
#       max_seq_len: int,
#       max_batch_size: int
#   ) -> LLaMA:
#       tokenizer_path = os.path.join(config.llama, "checkpoints",  "tokenizer.model")

#       start_time = time.time()
#       checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
#       assert world_size == len(
#           checkpoints
#       ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
#       ckpt_path = checkpoints[local_rank]
#       print("Loading")
#       checkpoint = torch.load(ckpt_path, map_location="cpu")
#       with open(Path(ckpt_dir) / "params.json", "r") as f:
#           params = json.loads(f.read())

#       model_args: ModelArgs = ModelArgs(
#           max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
#       )
#       tokenizer = Tokenizer(model_path=tokenizer_path)
#       model_args.vocab_size = tokenizer.n_words
#       torch.set_default_tensor_type(torch.cuda.HalfTensor)
#       model = Transformer(model_args)
#       torch.set_default_tensor_type(torch.FloatTensor)
#       model.load_state_dict(checkpoint, strict=False)

#       generator = LLaMA(model, tokenizer)
#       print(f"Loaded in {time.time() - start_time:.2f} seconds")
#       return generator

def predict_batch(
    tokenizer,
    model,
    test_example:dict,
    prompts,
    train_data,
    dataset:str,
    max_generated_len:int,
    batch_size: int,
):
    train_ids = [tuple(ids) for ids in prompts]
    output_text = []

    input_text = [format_prompt(test_example,
                                [train_data[train_id] for train_id in prompt],
                                dataset) for prompt in prompts]
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

        output_text.extend(tokenizer.batch_decode(outputs.sequences[:, -max_generated_len:]))


    predictions = {}
    for i, train_ids in enumerate(train_ids):
        predictions[str(tuple(train_ids))] = {
                "input_text": input_text[i],
                "output_text": output_text[i],
                "prediction": extract_prediction(output_text[i], dataset),
                "label": test_example["label"]
            }

    return predictions

def batch_inference(model, tokenizer, prompts, batch_size, mask_bos):
    output_tokens = torch.empty(0, dtype=torch.int64).to('cuda:0')
    first_token_scores = torch.empty(0, dtype=torch.float16).to('cuda:0')

    num_batches = ceil(len(prompts) / batch_size)

    for batch in range(num_batches):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(prompts))

        gen_len=5
        # tokenize by batch to mitigate effect of long outliers
        tokens = tokenizer(prompts[start:end], padding=True, return_tensors="pt").to('cuda:0')
        attention_mask = masked_bos(tokens.attention_mask) if mask_bos else tokens.attention_mask

        with torch.no_grad():
            outputs = model.generate(
                input_ids=tokens.input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_len,
                temperature=0,
                return_dict_in_generate=True,
                output_scores=True,
            )
        output_tokens = torch.cat((output_tokens, outputs.sequences[:, -gen_len:]))
        first_token_scores = torch.cat((first_token_scores, outputs.scores[0]))

    return output_tokens, first_token_scores

def batch_scoring(model, tokenizer, prompts, batch_size, mask_bos):
    log_probs = torch.empty(0, dtype=torch.float32).to('cpu:0')

    num_batches = ceil(len(prompts) / batch_size)
    with torch.no_grad():
        for batch in range(num_batches):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, len(prompts))

            # tokenize by batch to mitigate effect of long outliers
            tokens = tokenizer(prompts[start:end], padding=True, return_tensors="pt").to('cuda:0')
            attention_mask = masked_bos(tokens.attention_mask) if mask_bos else tokens.attention_mask
            logits = model(tokens.input_ids, attention_mask).logits
            labels_attention_mask = attention_mask.unsqueeze(-1)
            masked_log_probs = labels_attention_mask.float() * torch.log_softmax(
                logits.float(), dim=-1
            )
            seq_token_log_probs = torch.gather(
                masked_log_probs, -1, tokens.input_ids.unsqueeze(-1)
            )
            seq_token_log_probs = seq_token_log_probs.squeeze(dim=-1)
            seq_log_prob = seq_token_log_probs.sum(dim=-1).to("cpu")

            log_probs = torch.cat((log_probs, seq_log_prob))

    return log_probs

def masked_bos(a: torch.Tensor)->torch.Tensor:
    a[:, -1] = 0
    return torch.roll(a, shifts=1, dims=-1)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--experiment_id', type=int)
    parser.add_argument('--dataset', choices=config.tasks)
    parser.add_argument('--generation_id', type=int)
    parser.add_argument('--model', type=str.lower)
    parser.add_argument('--test-ids', type=int, nargs='+')
    parser.add_argument('-mask-bos', action='store_true')
    parser.add_argument('-f', action='store_true')

#   parser.add_argument('--uuid', type=str)
#SLURM_ARRAY_TASK_ID = os.getenv('SLURM_ARRAY_TASK_ID')


    args = parser.parse_args()
    dataset, exp_id, gen_id, model_name, test_ids = args.dataset, args.experiment_id, args.generation_id, args.model, args.test_ids
    assert None not in [dataset, exp_id, gen_id, model_name, test_ids]


#   if args.uuid:
#       log = get_log_with_uuid(args.uuid)
#       experiment_id = log['experiment_id']
#       generation_id = log['generation_id']

#   test_ids = [exp_info['test_ids'][i] for i in
#                   range(int(SLURM_ARRAY_TASK_ID),
#                       min(int(SLURM_ARRAY_TASK_ID) + config.tests_per_gpu,
#                       len(exp_info['test_ids'])))]

    exp_dir = get_dir_with_id(os.path.join(config.experiments, dataset), exp_id)
    train_data = read_jsonl(os.path.join(exp_dir, 'train.jsonl'))
    test_data = read_jsonl(os.path.join(exp_dir, 'test.jsonl'))

    train_data = {ex["idx"]: ex for ex in train_data}
    test_data = {ex["idx"]: ex for ex in test_data}

    test_examples = {str(i): test_data[i] for i in test_ids if i in test_data}

    models = {"opt-125m": "facebook/opt-125m",
              "opt-350m": "facebook/opt-350m",
              "opt-1.3b": "facebook/opt-1.3B",
              "opt-2.7b": "facebook/opt-2.7B",
              "opt-6.7b": "facebook/opt-6.7B",
              "llama-7b": "decapoda-research/llama-7b-hf",
              "alpaca-7b": "decapoda-research/llama-7b-hf"}

    lora_weights = {"alpaca-7b": "tloen/alpaca-lora-7b"}

    model_name = args.model
    model_addr = models[model_name]
    max_context_len = 2048

    batch_sizes = {"opt-125m": 64,
              "opt-350m": 32,
              "opt-1.3b": 16,
              "opt-2.7b": 8,
              "opt-6.7b": 4,
              "llama-7b": 4,
              "alpaca-7b": 4}
    batch_size = batch_sizes[args.model]

    print(f"Load {args.model}...", end="")
    model = None
    tokenizer = AutoTokenizer.from_pretrained(model_addr, use_fast=False, padding_side='left')
    if "llama" in model_name or "alpaca" in model_name:
        model = LlamaForCausalLM.from_pretrained(
            model_addr,
            torch_dtype=torch.float16,
            device_map='auto'
        ).cuda()
        if model_name in lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights[model_name],
                torch_dtype=torch.float16,
            )
            args.mask_bos=True
        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        model.half()
        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_addr).cuda()
    print("done!")


    generation_dir = get_dir_with_id(exp_dir, gen_id)
    model_dir = os.path.join(generation_dir, model_name)
    if args.mask_bos:
        model_dir = os.path.join(generation_dir, f"{model_name}_masked_bos")
    os.makedirs(model_dir, exist_ok=True)


    print("Get prompt_map...", end="")
    prompt_map_filepath = os.path.join(generation_dir, "prompt_map.json")
    with open(prompt_map_filepath, "r") as f:
        prompt_map = json.load(f)
    print("done!")


    missed = []
    # make predictions
    for test_id, example in tqdm(test_examples.items()):
#       try:
        predictions_folder= os.path.join(model_dir, test_id)
        predictions_filepath = os.path.join(predictions_folder, "predictions.json")
        predictions = {}

        if (not args.f) and os.path.exists(predictions_filepath):
            print(f"Example {test_id} already has predictions")
            continue

        if dataset in ["BoolQ", "CB", "RTE", "WiC", "WSC", "Winograd"]:
            prompts = []
            for train_ids in prompt_map[test_id]:
                key = config.delim.join([str(x) for x in train_ids])
                predictions[key] = {}

                prompt = format_few_shot([train_data[idx] for idx in train_ids], example, dataset)

                predictions[key]['prompt'] = prompt
                prompts.append(prompt)

            output_sequences, output_scores = batch_inference(model, tokenizer, prompts, batch_size, args.mask_bos)
            output_text = tokenizer.batch_decode(output_sequences)

            for i, train_ids in enumerate(prompt_map[test_id]):
                key = config.delim.join([str(x) for x in train_ids])
                predictions[key]['output_text'] = output_text[i]
                predictions[key]['probs'] = first_token_probs(output_scores[i], dataset)
                predictions[key]['prediction'] = verbalize(output_text[i], dataset)

        else:
            p_map = {}
            for train_ids in prompt_map[test_id]:
                key = config.delim.join([str(x) for x in train_ids])
                predictions[key] = {}

                prompt = format_few_shot([train_data[idx] for idx in train_ids], example, dataset)
                predictions[key]['prompt'] = prompt
                choices = format_few_shot_choices([train_data[idx] for idx in train_ids], example, dataset)

                choices = {config.delim.join([key, label]): choice for label, choice in choices.items()}
                p_map.update(choices)

            idxs, prompts = zip(*p_map.items())
            probs = batch_scoring(model, tokenizer, prompts, batch_size, args.mask_bos)
            results = pack(idxs, probs, dataset)
            choices = verbalize(results, dataset)

            for train_ids in prompt_map[test_id]:
                key = config.delim.join([str(x) for x in train_ids])
                predictions[key]["log_probs"] = results[key]
                predictions[key]["prediction"] = choices[key]


        os.makedirs(predictions_folder, exist_ok=True)
        # save outputs and monitor information
        with open(predictions_filepath, "w") as f:
            json.dump(predictions, f, indent=4)

#       except Exception as e:
#       print(
#           f"Example {test_id} has some problem."
#       )
#       print(e)
#       missed.append(test_id)

    print("All finished.")
    if len(missed) > 0:
        print("Missed:", ", ".join([str(x) for x in missed]))

if __name__ == "__main__":
    main()
