#!/bin/bash
shopt -s expand_aliases #for python3 alias
alias python='python3' #remove if necessary

#allows running script from location in interactive shell (assuming all scripts are in /scripts directory)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DATA_HOME=$SCRIPT_DIR/../data
SRC_HOME=$SCRIPT_DIR/../src
EXPERIMENTS_HOME=$SCRIPT_DIR/../experiments

export PYTHONPATH=.