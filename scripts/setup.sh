#!/bin/bash
set -e

dataset=${1:-"BoolQ"}
train=${2:-8}
test=${3:-8}
uuid=${4:-""}

to_gpu python $SRC_HOME/setup.py \
--dataset $dataset --train $train --test $test --uuid $uuid