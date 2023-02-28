#!/bin/bash -l
#SBATCH --job-name mice-inference
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/experiments/outputs/%a.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/experiments/outputs/%a.err
#SBATCH --gres=gpu:1
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --time 5
#SBATCH --requeue
#NOTE: specify --array when invoking sbatch
source ./config.sh

experiment_id=$1
generation_id=$2
model=$3
test_ids_path=$4

sid=$SLURM_ARRAY_TASK_ID
test_id=$(sed "${sid}q;d" $test_ids_path)

python $SRC_HOME/inference.py \
$experiment_id $generation_id $test_id --model $model