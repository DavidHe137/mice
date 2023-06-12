#!/usr/bin/env python3
#SBATCH --job-name mice-inference
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/logs/inference/%A.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/logs/inference/%A.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task 6
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --time 06:00:00
#SBATCH --requeue

import os
import sys
import json
import argparse
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from tqdm import tqdm
from math import ceil, floor, log

sys.path.append("/coc/pskynet6/dhe83/mice/src")
from utils import *
from prompts import *
import time

greedy = GenerationConfig(
    max_new_tokens=5,
    temperature=0,
    return_dict_in_generate=True,
    output_scores=True,
)

sampling = GenerationConfig(
    max_new_tokens=200,
    temperature=0.5,
    do_sample=True,
    top_k=40,
    return_dict_in_generate=True,
    output_scores=True,
)

available_bytes = torch.cuda.mem_get_info()[0]
# TODO: dynamic batch size
def maximum_batch_size(available_bytes, max_tokens):
    per_token = 2 * 2**20
    batch_size =  available_bytes // (per_token * max_tokens)
    batch_size = 2**(floor(log(batch_size, 2)))
    return batch_size

def max_token_len(tokenizer, prompts, gen_len=0):
    return max([len(tokenizer(p).input_ids) for p in prompts]) + gen_len

def generate_single(model, tokens, attention_mask):
    outputs = model.generate(
        input_ids=tokens.input_ids,
        attention_mask=attention_mask,
        generation_config=gen_config
        )
    return outputs

def self_consistency(model, tokens, attention_mask, num_paths):
    for _ in num_paths:
        outputs = model.generate(
            input_ids=tokens.input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config
        )

def batch_inference(model, tokenizer, prompts, gen_config, mask_bos, self_consistency):
    output_tokens = torch.empty(0, dtype=torch.int64).to('cuda:0')
    first_token_scores = torch.empty(0, dtype=torch.float16).to('cuda:0')

    gen_len = gen_config.max_new_tokens
    batch_size = maximum_batch_size(available_bytes, max_token_len(tokenizer, prompts, gen_len))
    num_batches = ceil(len(prompts) / batch_size)

    for batch in range(num_batches):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(prompts))

        # tokenize by batch to mitigate effect of long outliers
        tokens = tokenizer(prompts[start:end], padding=True, return_tensors="pt").to('cuda:0')
        attention_mask = masked_bos(tokens.attention_mask) if mask_bos else tokens.attention_mask

        #TODO: generation config
        with torch.no_grad():
            if self_consistency:

            else:
                outputs = model.generate(
                    input_ids=tokens.input_ids,
                    attention_mask=attention_mask,
                    generation_config=gen_config
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
    parser.add_argument('-self-consistency', action='store_true')
    parser.add_argument('-f', action='store_true')
#   parser.add_argument('--uuid', type=str)


    args = parser.parse_args()
    dataset, exp_id, gen_id, model_name, test_ids = args.dataset, args.experiment_id, args.generation_id, args.model, args.test_ids
    assert None not in [dataset, exp_id, gen_id, model_name, test_ids]


#   if args.uuid:
#       log = get_log_with_uuid(args.uuid)
#       experiment_id = log['experiment_id']
#       generation_id = log['generation_id']

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
              "llama-13b": "decapoda-research/llama-13b-hf",
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
              "llama-13b": 2,
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

    start = time.time()
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

        if dataset in ["BoolQ", "CB", "RTE", "WiC", "WSC", "Winograd", "GSM8K"]:
            prompts = []
            for train_ids in prompt_map[test_id]:
                key = config.delim.join([str(x) for x in train_ids])
                predictions[key] = {}

                prompt = format_few_shot([train_data[idx] for idx in train_ids], example, dataset)

                predictions[key]['prompt'] = prompt
                prompts.append(prompt)

            gen_config = greedy
            if dataset == "GSM8K":
                gen_config = sampling

            output_sequences, output_scores = batch_inference(model, tokenizer, prompts, gen_config, args.mask_bos, args.self_consistency)
            output_text = tokenizer.batch_decode(output_sequences)

            for i, train_ids in enumerate(prompt_map[test_id]):
                key = config.delim.join([str(x) for x in train_ids])
                predictions[key]['output_text'] = output_text[i]

                if dataset in ["BoolQ", "WSC", "WIC"]:
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

    end = time.time()
    print("Time elapsed:", end-start)
    print("Time/example:", (end-start)/float(len(test_examples)))
if __name__ == "__main__":
    main()
