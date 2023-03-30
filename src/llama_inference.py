import os
import json
import config
from utils import *

# llama imports
from pathlib import Path
from typing import Tuple
import sys
import fire
import torch
import time
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

sys.path.append(config.llama)
from llama import ModelArgs, Transformer, Tokenizer, LLaMA

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def load(
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


def main(
    model: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    ckpt_dir = os.path.join(config.llama, "checkpoints", model.upper())
    generator = load(
        ckpt_dir, local_rank, world_size, max_seq_len, max_batch_size
    )

    prompts = ["Text: Tom said \"Check\" to Ralph as he moved his bishop.\nQuestion: Does his refer to Ralph?\nAnswer:False\nText: Tom said \"Check\" to Ralph as he moved his bishop.\nQuestion: Does his refer to Ralph?\nAnswer:False\nText: Mr. Moncrieff visited Chester 's luxurious New York apartment, thinking that it belonged to his son Edward . The result was that Mr. Moncrieff has decided to cancel Edward 's allowance on the ground that he no longer requires his financial support.\nQuestion: Does his refer to Mr. Moncrieff?\nAnswer:",
               "Text: Bountiful arrived after war's end, sailing into San Francisco Bay 21 August 1945. Bountiful was then assigned as hospital ship at Yokosuka, Japan, departing San Francisco 1 November 1945.\nHypothesis: Bountiful reached San Francisco in August 1945.\nEntailment:entailment\nText: Bountiful arrived after war's end, sailing into San Francisco Bay 21 August 1945. Bountiful was then assigned as hospital ship at Yokosuka, Japan, departing San Francisco 1 November 1945.\nHypothesis: Bountiful reached San Francisco in August 1945.\nEntailment:entailment\nText: In 1979, the leaders signed the Egypt-Israel peace treaty on the White House lawn. Both President Begin and Sadat received the Nobel Peace Prize for their work. The two nations have enjoyed peaceful relations to this day.\nHypothesis: The Israel-Egypt Peace Agreement was signed in 1979.\nEntailment:"]
    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)