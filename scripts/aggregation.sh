#!/bin/bash
set -e
source ./config.sh

experiment_id=1
generation_id=1
model_size=350m
method=mice-sampling

to_gpu python ./src/aggregation.py \
$experiment_id $generation_id $model_size --method $method           