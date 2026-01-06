#!/usr/bin/env python3
#SBATCH --job-name mice-inference-ids
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/logs/inference/%A.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/logs/inference/%A.err
#SBATCH --partition=short
#SBATCH --account=short
#SBATCH --time 5

import os
import json
import argparse
import sys
sys.path.append("/coc/pskynet6/dhe83/mice/src")
import config
import subprocess
from utils import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment-id", type=int)
    parser.add_argument('--dataset', choices=config.tasks)
    parser.add_argument('--gpus', type=int)

    args = parser.parse_args()
    exp_dir = get_dir_with_id(os.path.join(config.experiments, args.dataset), args.experiment_id)

    with open(os.path.join(exp_dir, "summary.json"), "r") as f:
        a = json.load(f)
        print(" ".join([str(x) for x in a['test_ids']]))

    per_gpu = len(a['test_ids']) // args.gpus
    for i in range(0, len(a['test_ids']), per_gpu):
        ids = a['test_ids'][i: min(i+per_gpu, len(a['test_ids']))]
        print("Calling setup.py...")
        command = f'''sbatch /nethome/dhe83/mice/src/inference.py --dataset {args.dataset} --experiment-id {args.experiment_id} --generation-id 1 --test-ids {" ".join([str(x) for x in ids])} --model llama-7b'''
        print(command)
        setup = subprocess.run(command.split(),
                                stdout=subprocess.PIPE, text=True, check=True)
        print(setup.stdout)

    return
if __name__ == "__main__":
    main()

