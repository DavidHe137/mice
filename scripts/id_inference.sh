#!/bin/bash -l
source ./config.sh

experiment_id=1
generation_id=1
model_size=350m
test_id=$1

to_gpu python $SRC_HOME/inference.py \
$experiment_id $generation_id $test_id --model $model_size