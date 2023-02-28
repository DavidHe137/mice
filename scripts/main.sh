#!/bin/bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/config.sh

dataset=BoolQ
train=8
test=64

generation=similar
in_context=2
max_num_prompts=16

model_size=125m

method=mice_sampling

to_gpu python $SRC_HOME/setup.py \
--dataset $dataset --train $train --test $test

to_gpu python $SRC_HOME/prompt_generation.py \
1 --method $generation --in_context $in_context --max_num_prompts $max_num_prompts

task_ids=$(wc -l $EXPERIMENTS_HOME/BoolQ/id_1_train_8_test_64/test_ids.txt | awk '{ print $1 }')
sbatch --wait --array=$task_ids%8 $SRC_HOME/batch_inference
wait

python $SRC_HOME/aggregation.py
1 1 26