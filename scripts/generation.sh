#!/bin/bash
#SBATCH --job-name mice-generation
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/outputs/generation_%A.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/outputs/generation_%A.err
#SBATCH --gres=gpu:1
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --time 5
#SBATCH --requeue
set -e
source ./config.sh

experiment_id=${1:-"1"}
generation=${2:-"similar"}
in_context=${3:-2}
max_num_prompts=${4:-16}
uuid=${5:-""}

to_gpu python $SRC_HOME/prompt_generation.py \
$experiment_id --generation $generation --in_context $in_context --max_num_prompts $max_num_prompts --uuid $uuid