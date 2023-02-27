#!/bin/bash -l
#SBATCH --job-name mice
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/experiments/outputs/%a.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/experiments/outputs/%a.err
#SBATCH --gres=gpu:1
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --time 5
#SBATCH --array=1-64
#SBATCH --requeue
source ./config.sh

experiment_id=1
generation_id=1
model_size=350m


sid=$SLURM_ARRAY_TASK_ID
test_id=$(sed "${sid}q;d" $EXPERIMENTS_HOME/BoolQ/id_1_train_8_test_64/test_ids.txt)

python $SRC_HOME/inference.py \
$experiment_id $generation_id $test_id --model $model_size