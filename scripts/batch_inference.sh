#!/bin/bash -l
#SBATCH --job-name mice-inference
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/outputs/inference/%a.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/outputs/inference/%a.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task 6
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --time 15
#SBATCH --requeue
#NOTE: time should depend on model size
#NOTE: specify --array, --constraint when invoking sbatch

experiment_id=$1
generation_id=$2
model=$3
uuid=${4:-""}
if [[ ! -z "$uuid" ]]; then 
  uuid="--uuid $uuid" 
fi

cmd=python
if [[ $model == *"llama"* ]]; then
  cmd="torchrun --nproc_per_node 1"
fi

#FIXME: hardcoded path
$cmd /coc/pskynet6/dhe83/mice/src/inference.py \
$experiment_id $generation_id $SLURM_ARRAY_TASK_ID $model $uuid