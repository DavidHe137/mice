#!/bin/bash
shopt -s expand_aliases

#allows running script from location in interactive shell (assuming all scripts are in /scripts directory)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DATA_HOME=$SCRIPT_DIR/../data
SRC_HOME=$SCRIPT_DIR/../src
EXPERIMENTS_HOME=$SCRIPT_DIR/../experiments

alias to_gpu="srun --gpus-per-node=1 -J \"mice\" -A overcap -p overcap"  

eval "$(conda shell.bash hook)"
conda activate mice

export PYTHONPATH=.