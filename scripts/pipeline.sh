#!/bin/bash
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/logs/slurm/pipeline_%A.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/logs/slurm/pipeline_%A.err
set -e
source ./config.sh

new=true
dataset=BoolQ
train=4
test=16
max_num_prompts=8
model=350m
method=mice-sampling

python $SRC_HOME/pipeline.py \
--dataset $dataset --train $train --test $test --max_num_prompts $max_num_prompts --model $model --method $method