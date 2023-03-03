#!/bin/bash -l
#SBATCH --job-name mice-inference
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/outputs/%a.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/outputs/%a.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=titan_x|rtx_6000|a40
#SBATCH --cpus-per-task 6
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --time 15
#SBATCH --requeue
#NOTE: time should depend on model size
#NOTE: specify --array when invoking sbatch

experiment_id=$1
generation_id=$2
model=$3
test_ids_path=$4
uuid=${5:-""}

sid=$SLURM_ARRAY_TASK_ID
test_id=$(sed "${sid}q;d" $test_ids_path)

#FIXME: hardcoded path
python /coc/pskynet6/dhe83/mice/src/inference.py \
$experiment_id $generation_id $test_id --model $model --uuid $uuid