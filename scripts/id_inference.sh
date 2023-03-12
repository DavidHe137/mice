#!/bin/bash -l
source ./config.sh

experiment_id=${1:-0}
generation_id=${2:-0}
model=${3:-"opt-125m"}
test_id=${4:-0}

cmd=python

if [[ $model == *"llama"* ]]; then
  cmd="torchrun --nproc_per_node 1"
fi

to_gpu $cmd $SRC_HOME/inference.py \
$experiment_id $generation_id 0 $model --test_id $test_id