#!/bin/bash
set -e
source ./config.sh

experiment_id=1
generation_id=1
model_size=350m
method=${1:-"mice-sampling"}

to_gpu python $SRC_HOME/aggregation.py \
$experiment_id $generation_id $model_size --method $method           