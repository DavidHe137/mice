import os
import json
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import *

def create_model_input(
    test_example, train_examples, dataset, tokenizer
):
    #TODO: worry about input len at some point
    demonstrations = ''.join(list(map(lambda x: format_example(x, dataset, includeLabel=True), train_examples)))
    test_input = format_example(test_example, dataset)

    input_text = demonstrations + test_input
    input_tokens = tokenizer.encode(input_text, return_tensors="pt").cuda()

    return {"input_text": input_text, "input_tokens": input_tokens}


def predict_with_gptj(
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
            input = create_model_input(
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
                "input_text": input["input_text"],
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


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment_id', type=str)
    parser.add_argument('generation_id', type=str)
    parser.add_argument('test_id', type=str)
    parser.add_argument('--model', default="125m", type=str)
    parser.add_argument('--uuid', type=str)

    args = parser.parse_args()

    experiment_id = args.experiment_id
    generation_id = args.generation_id
    if args.uuid:
        log = get_log_with_uuid(args.uuid)
        experiment_id = str(log['experiment_id'])
        generation_id = str(log['generation_id'])

    exp_info = get_experiment_info(experiment_id)

    train_data = read_jsonl(os.path.join(exp_info['location'], 'train.jsonl'))
    test_data = read_jsonl(os.path.join(exp_info['location'], 'test.jsonl'))

    train_data = {ex["idx"]: ex for ex in train_data}
    test_data = {ex["idx"]: ex for ex in test_data}

    test_example = test_data[int(args.test_id)]
    assert test_example

    models = {"125m": "facebook/opt-125m",
              "350m": "facebook/opt-350m",
              "1.3b": "facebook/opt-1.3B"}

    print("Load model...", end="")
    model_name = models[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    max_generated_len = 16 #NOTE: for SuperGLUE
    max_context_len = 2048
    print("done!")

    generation_dir = exp_info['generations'][generation_id]['location']
    example_dir = os.path.join(generation_dir, args.model, args.test_id)
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
        predictions, end_prediction = predict_with_gptj(
            exp_info['dataset'],
            test_example,
            train_data,
            prompt_map[args.test_id],
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
