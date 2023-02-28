#!/bin/bash
#SBATCH --job-name mice-aggregation
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/experiments/outputs/%a.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/experiments/outputs/%a.err
#SBATCH --gres=gpu:1
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --time 5
#SBATCH --requeue
set -e
source ./config.sh

experiment_id=$1
generation_id=$2
model_size=$3
method=$4
uuid=$5

to_gpu python $SRC_HOME/aggregation.py \
$experiment_id $generation_id $model_size --method $method --uuid $uuid        