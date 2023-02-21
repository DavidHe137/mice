#!/bin/bash
set -e
source ./config.sh

dataset=BoolQ
train=8
test=64

generation=similar
in_context=2
max_num_prompts=16

model_size=125m

method=mice_sampling

python ./src/experiment_setup.py \
--dataset $dataset --train $train --test $test

python ./src/prompt_generation.py \
1 --method $generation --in_context $in_context --max_num_prompts $max_num_prompts

python ./src/inference.py \
1 1 26