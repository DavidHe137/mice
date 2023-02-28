'''
#if new false, search(return error if not found), otherwise create new
new=false
dataset=BoolQ
train=8
test=64
#experiment id

#search to see if geneeration exists with exact same parameters
generation=similar
in_context=2
max_num_prompts=16
#generation_id

#check to see if folder exists
model_size=125m

#check to see if file exists
method=mice_sampling
'''

import subprocess
import argparse
import readline
import sys
import os
import re
from pathlib import Path
from uuid import uuid4
from utils import *

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--new", action="store_true", help="Create a new train/test")

    #setup.py
    parser.add_argument('--dataset', choices=['BoolQ'])
    parser.add_argument('--train', type=int)
    parser.add_argument('--test', type=int)

    #prompt_generation.py
    parser.add_argument('--generation', default='similar', choices=['similar', 'random', 'bayesian_noise'])
    parser.add_argument('--in_context', default=2, type=int)
    parser.add_argument('--max_num_prompts', default=1, type=int)
    parser.add_argument('--encoder', default='all-roberta-large-v1', type=str)

    #inference.py
    parser.add_argument('--model', default="125m", type=str)

    #aggregation.py
    parser.add_argument('--method', default="mice-sampling", choices=['mice-sampling', 'majority-vote'], type=str)

    args = parser.parse_args()
    
    if not args.interactive:
        run(args)
        return
    
    
    # First, keep ArgParser from exiting on invalid input
    class InvalidArgs(Exception):
        pass
    def exit(*args, **kwargs):
        raise InvalidArgs
    parser.exit = exit

    print("Enter commands. Use 'help' for info, 'exit' to leave.")
    while True:
        try:
            command = input('> ').strip()
            
            # remove the program name if they typed it
            prog_name = os.path.basename(sys.argv[0])
            command = re.sub(r'^{}\s+'.format(prog_name), '', command)
        except KeyboardInterrupt:
            # Ctrl-c clears the input
            sys.stdout.write('\n')
            continue
        except EOFError:
            # Ctrl-d exits
            sys.stdout.write('\n')
            break

        if command == 'exit':
            break

        if command in ['help', 'h', '?']:
            parser.print_help()
            continue

        try:
            command_args = parser.parse_args(args=command.split())
        except InvalidArgs:
            print('Invalid command')
            continue

        if command_args.interactive:
            # Don't let them get clever
            continue

        run(command_args)

def slurm_job_id(job: subprocess.CompletedProcess[str]) -> str:
    return job.stdout.strip().split(" ")[-1]

def run(args):
    project_root = Path(__file__).resolve().parents[1]
    scripts = os.path.join(project_root, 'scripts')

    uuid = str(uuid4())
    log_file = os.path.join(project_root, 'logs', f"{uuid}.json")


    #Setup
    setup_cmd = f'''sbatch {scripts}/setup.sh {args.dataset} {args.train} {args.test} {uuid}'''

    print("Calling setup.py...", end="")
    setup = subprocess.run(setup_cmd.split(" "), 
                            stdout=subprocess.PIPE, text=True, check=True)
    print(setup.stdout)
    log = read_json(log_file)


    #Prompt Generation
    prompt_generation_cmd = f'''sbatch {scripts}/prompt_generation.sh {log['experiment_id']} {args.generation} {args.in_context} {args.max_num_prompts} {uuid}'''

    print("Calling prompt_generation.py...", end="")
    prompt_generation = subprocess.run(prompt_generation_cmd.split(" "), 
                            stdout=subprocess.PIPE, text=True, check=True)
    print(prompt_generation.stdout)
    log = read_json(log_file)


    #Inference
    exp_summary = os.path.join(project_root, 'experiments', 'summary.json')
    exp_summary_data = read_json(exp_summary)
    assert str(log['experiment_id']) in exp_summary_data
    test_ids_path= os.path.join(exp_summary_data[str(log['experiment_id'])]['location'], 'test_ids.txt')

    inference_cmd = f'''sbatch --array=1-{args.test} {scripts}/batch_inference.sh {log['experiment_id']} {log['generation_id']} {args.model} {test_ids_path}'''

    print("Queueing inference job array...", end="")
    inference = subprocess.run(inference_cmd.split(" "), 
                            stdout=subprocess.PIPE, text=True, check=True)
    print(inference.stdout)
    inference_id = slurm_job_id(inference)


    #Aggregation
    aggregation_cmd = f'''sbatch --dependency=afterok:{inference_id} {scripts}/aggregation.sh {log['experiment_id']} {log['generation_id']} {args.model} {args.method} {uuid}'''

    print("Batching aggregation.py...", end="")
    aggregation = subprocess.run(aggregation_cmd.split(" "), 
                            stdout=subprocess.PIPE, text=True, check=True)
    print(aggregation.stdout)
    print("All jobs queued.")

def validate(args):
    print("validate")


if __name__ == "__main__":
    parse()