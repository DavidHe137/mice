#!/bin/bash
#SBATCH --job-name mice-setup
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/experiments/outputs/%Asetup.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/experiments/outputs/%Asetup.err
#SBATCH --gres=gpu:1
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --time 5
#SBATCH --requeue
set -e
source ./config.sh

dataset=${1:-"BoolQ"}
train=${2:-8}
test=${3:-8}
uuid=${4:-""}

to_gpu python $SRC_HOME/setup.py \
--dataset $dataset --train $train --test $test --uuid $uuid