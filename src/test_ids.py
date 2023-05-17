import os
import json
import argparse
import sys
sys.path.append("/coc/pskynet6/dhe83/mice/src")
import config
from utils import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_id", type=int)
    parser.add_argument('--dataset', choices=config.tasks)

    args = parser.parse_args()
    exp_dir = get_dir_with_id(os.path.join(config.experiments, args.dataset), args.experiment_id)

    with open(os.path.join(exp_dir, "summary.json"), "r") as f:
        a = json.load(f)
        print(" ".join([str(x) for x in a['test_ids']]))
    return
if __name__ == "__main__":
    main()

