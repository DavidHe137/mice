#!/bin/bash
#SBATCH --job-name mice-test
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/%A.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 6
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --time 5
#SBATCH --requeue
uuid=${1:-""}
if [[ ! -z "$uuid" ]]; then 
  uuid="--uuid $uuid" 
fi

echo $uuid