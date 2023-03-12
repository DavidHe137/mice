#!/usr/bin/env python3
#SBATCH --job-name mice-pipeline
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/logs/slurm/%A.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/logs/slurm/%A.err
#SBATCH --gres=gpu:1
#SBATCH --partition=overcap
#SBATCH --account=overcap
#NOTE: should this be 10?
#SBATCH --time 10
#SBATCH --requeue

import sys
import os

import subprocess
import argparse
import re
from uuid import uuid4

sys.path.append(os.getcwd()) 
from utils import *
import config

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    #setup.py
    parser.add_argument("--experiment_id", type=str)
    parser.add_argument('--dataset', choices=['BoolQ', 'COPA', 'RTE', 'WiC', 'WSC'])
    parser.add_argument('--train', type=int)
    parser.add_argument('--test', type=int)

    #prompt_generation.py
    parser.add_argument('--ordering', default='similar', choices=['similar', 'random', 'bayesian_noise'])
    parser.add_argument('--in_context', default=2, type=int)
    parser.add_argument('--max_num_prompts', default=1, type=int)
    parser.add_argument('--encoder', default='all-roberta-large-v1', type=str)

    #inference.py
    parser.add_argument('--model', default="opt-125m", type=str)

    #aggregation.py
    parser.add_argument('--method', default="mice-sampling", choices=['mice-sampling', 'majority-vote'], type=str)

    args = parser.parse_args()
    
    if not args.interactive:
        if args.experiment_id:
            run_old(args)
        else:
            run_clean(args)
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
        
        run_clean(command_args)

def slurm_job_id(job: subprocess.CompletedProcess[str]) -> str:
    return job.stdout.strip().split(" ")[-1]

def prompt_generation(ordering, in_context, max_num_prompts, uuid):
    print("Calling prompt_generation.py...")
    command = f'''python {config.src}/prompt_generation.py 0 --ordering {ordering} --in_context {in_context} --max_num_prompts {max_num_prompts} --uuid {uuid}'''
    print(command)
    prompt_generation = subprocess.run(command.split(" "), 
                            stdout=subprocess.PIPE, text=True, check=True)
    print(prompt_generation.stdout)

def setup(dataset, train, test, uuid):
    print("Calling setup.py...")
    command = f'''python {config.src}/setup.py --dataset {dataset} --train {train} --test {test} --uuid {uuid}'''
    print(command)
    setup = subprocess.run(command.split(" "), 
                            stdout=subprocess.PIPE, text=True, check=True)
    print(setup.stdout)

def inference(test, model, uuid) -> str:
    # NOTE: Returns SLURM job id.
    constraint = "titan_x|rtx_6000|a40"
    if "llama" in model:
        constraint = "a40"

    print("Queueing inference job array...")
    command = f'''sbatch --array=0-{test} --constraint={constraint} {config.scripts}/batch_inference.sh 0 0 {model} {uuid} '''
    print(command)
    inference = subprocess.run(command.split(" "), 
                            stdout=subprocess.PIPE, text=True, check=True)
    print(inference.stdout)
    return f":{slurm_job_id(inference)}"

def aggregation(dependencies, model, method, uuid):
    dependency_arg = ""
    if dependencies:
        dependency_arg = f"--dependency=afterany{dependencies}"

    print("Batching aggregation.py...")
    command = f'''sbatch {dependency_arg} {config.scripts}/aggregation.sh 0 0 {model} {method} {uuid}'''
    print(command)
    aggregation = subprocess.run(command.split(" "), 
                            stdout=subprocess.PIPE, text=True, check=True)
    print(aggregation.stdout)

def run_old(args):    
    uuid = str(uuid4())
    print("Run:", uuid)

    exp_data = get_experiment_info(args.experiment_id)

    print("Existing experiment found:")
    print('Dataset', exp_data['dataset'])
    print('Train', exp_data['train'])
    print('Test', exp_data['test'])
    
    #check if generation exists
    generation_id = None
    for k, v in exp_data['generations'].items():
        if (v['ordering'] == args.ordering and 
            v['in_context'] == args.in_context and 
            v['max_num_prompts'] == args.max_num_prompts and 
            v['encoder'] == args.encoder):
            generation_id = k
            break

    log = {'uuid': uuid, 'experiment_id': args.experiment_id, 'status': 'prompt_generation'}
    if generation_id:
        print("Found generation id:", generation_id)
        log['generation_id'] = int(generation_id)
        log['status'] = 'inference'
    write_json(log, os.path.join(config.logs, f"{uuid}.json"))

    # prompt generation
    if not generation_id:
        prompt_generation(args.ordering, args.in_context, args.max_num_prompts, args.uuid)

    # inference
    dependencies = ""
    if not(generation_id and os.path.exists(os.path.join(exp_data['generations'][generation_id]['location'], args.model))):
        dependencies = inference(args.test, args.model, uuid)

    else:
        print("Found predictions for", args.model)
    
    aggregation(dependencies, args.model, args.method, uuid)
    print("All jobs queued.")


def run_clean(args):
    uuid = str(uuid4())
    print("Run:", uuid)

    # sample data, create folders, initialize logging
    setup(args.dataset, args.train, args.test, uuid)

    # generate prompt_map.json
    prompt_generation(args.ordering, args.in_context, args.max_num_prompts, uuid)

    # run inference 
    dependencies = inference(args.test, args.model, uuid)

    # report performance
    aggregation(dependencies, args.model, args.method, uuid)

    print("All jobs queued.")
  

def validate(args):
    print("validate")

if __name__ == "__main__":
    os.makedirs(config.experiments, exist_ok=True)
    os.makedirs(os.path.join(config.outputs, "aggregation"), exist_ok=True)
    os.makedirs(os.path.join(config.outputs, "inference"), exist_ok=True)
    os.makedirs(os.path.join(config.logs, "slurm"), exist_ok=True)
    parse()