#!/bin/bash
set -e
source ./config.sh

dataset=BoolQ
train=8
test=64

to_gpu python $SRC_HOME/experiment_setup.py \
--dataset $dataset --train $train --test $test