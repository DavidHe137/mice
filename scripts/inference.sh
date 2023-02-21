#!/bin/bash -l
#SBATCH --job-name mice
#SBATCH --output=$EXPERIMENTS_HOME/outputs/%a.out
#SBATCH --error=$EXPERIMENTS_HOME/outputs/%a.err
#SBATCH --gres=gpu:1
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH -a 1-251
#SBATCH --requeue
source ./config.sh

experiment_id=1
generation_id=1
model_size=350m

sid=$SLURM_ARRAY_TASK_ID
test_id=$(sed "${sid}q;d" $EXPERIMENTS_HOME/BoolQ/id_1_train_8_test_64/test_ids.txt)

python $SRC_HOME/inference.py \
$experiment_id $generation_id $test_id --model $model_size