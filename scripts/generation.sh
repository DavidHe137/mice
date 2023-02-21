#!/bin/bash
set -e
source ./config.sh

experiment_id=1
generation=similar
in_context=2
max_num_prompts=16

to_gpu python $SRC_HOME/prompt_generation.py \
$experiment_id --generation $generation --in_context $in_context --max_num_prompts $max_num_prompts