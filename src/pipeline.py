#!/usr/bin/env python3
#SBATCH --job-name mice-pipeline
#SBATCH --output=/srv/nlprx-lab/share6/dhe83/mice/logs/pipeline/%A.out
#SBATCH --error=/srv/nlprx-lab/share6/dhe83/mice/logs/pipeline/%A.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --requeue

import sys
import os

import subprocess
import argparse
from uuid import uuid4

sys.path.append("/coc/pskynet6/dhe83/mice/src")
from utils import *
import config

def parse():
    parser = argparse.ArgumentParser()

    #setup.py
    parser.add_argument("--experiment_id", type=int)
    parser.add_argument('--dataset', choices=config.tasks)
    parser.add_argument('--train', type=int)
    parser.add_argument('--test', type=int)

    #prompt_generation.py
    parser.add_argument('--ordering', default='similar', choices=['similar', 'random', 'bayesian_noise'])
    parser.add_argument('--in_context', default=2, type=int)
    parser.add_argument('--max_num_prompts', default=1, type=int)

    #inference.py
    parser.add_argument('--model', default="opt-125m", type=str)

    #aggregation.py
    parser.add_argument('--method', default="mice-sampling", choices=['mice-sampling', 'majority-vote'], type=str)

    args = parser.parse_args()

    if args.experiment_id:
        run_existing(args)
    else:
        run_clean(args)

def slurm_job_id(job: subprocess.CompletedProcess[str]) -> str:
    return job.stdout.strip().split(" ")[-1]

def format_dependencies(jobs):
    if not isinstance(jobs, list):
        jobs = [jobs]
    return f"--dependency=afterany{''.join([f':{id}' for id in jobs])}"

def setup(dataset, train, test, uuid, dependencies = []):
    dependencies = format_dependencies(dependencies) if dependencies else ''

    print("Calling setup.py...")
    command = f'''sbatch {dependencies} {config.src}/setup.py --dataset {dataset} --train {train} --test {test} --uuid {uuid}'''
    print(command)
    setup = subprocess.run(command.split(),
                            stdout=subprocess.PIPE, text=True, check=True)
    print(setup.stdout)
    return slurm_job_id(setup)

def prompt_generation(ordering, in_context, max_num_prompts, uuid, dependencies=[]):
    dependencies = format_dependencies(dependencies) if dependencies else ""

    print("Calling prompt_generation.py...")
    command = f'''sbatch {dependencies} {config.src}/prompt_generation.py 0 --ordering {ordering} --in_context {in_context} --max_num_prompts {max_num_prompts} --uuid {uuid}'''
    print(command)
    prompt_generation = subprocess.run(command.split(),
                            stdout=subprocess.PIPE, text=True, check=True)
    print(prompt_generation.stdout)
    return slurm_job_id(prompt_generation)

def inference(test, model, uuid, dependencies=[]) -> str:
    # FIXME: can't rely on test if testing whole dataset
    dependencies = format_dependencies(dependencies) if dependencies else ""

    print("Queueing inference job array...")
    command = f'''sbatch {dependencies} --array=0-{test}:{config.tests_per_gpu} {config.src}/inference.py 0 0 {model} --uuid {uuid} '''
    print(command)
    inference = subprocess.run(command.split(),
                            stdout=subprocess.PIPE, text=True, check=True)
    print(inference.stdout)
    return slurm_job_id(inference)

def aggregation(model, method, uuid, dependencies=[]):
    dependencies = format_dependencies(dependencies) if dependencies else ""

    print("Batching aggregation.py...")
    command = f'''sbatch {dependencies} {config.src}/aggregation.py 0 0 {model} --method {method} --uuid {uuid}'''
    print(command)
    aggregation = subprocess.run(command.split(),
                            stdout=subprocess.PIPE, text=True, check=True)
    print(aggregation.stdout)
    return slurm_job_id(aggregation)

def run_existing(args):
    uuid = str(uuid4())
    print("Run:", uuid)

    print("Existing experiment found:")
    print('Dataset', exp_data['dataset'])
    print('Train', exp_data['train'])
    print('Test', exp_data['test'])

    test = exp_data['test']

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
        log['generation_id'] = generation_id
        log['status'] = 'inference'
    write_json(log, os.path.join(config.logs, f"{uuid}.json"))

    dependencies = []
    # prompt generation
    if not generation_id:
        generation_job = prompt_generation(args.ordering, args.in_context, args.max_num_prompts, args.uuid)
        dependencies.append(generation_job)

    #TODO: loop through all examples and check for existing
    # inference
    if not(generation_id and os.path.exists(os.path.join(exp_data['generations'][generation_id]['location'], args.model))):
        inference_job = inference(test, args.model, uuid, dependencies=dependencies)
        dependencies.append(inference_job)

    else:
        print("Found predictions for", args.model)

    aggregation(args.model, args.method, uuid, dependencies=dependencies)
    print("All jobs queued.")


def run_clean(args):
    uuid = str(uuid4())
    print("Run:", uuid)

    # sample data, create folders, initialize logging
    setup_job = setup(args.dataset, args.train, args.test, uuid)

    # generate prompt_map.json
    generation_job = prompt_generation(args.ordering, args.in_context, args.max_num_prompts, uuid, dependencies=setup_job)

    # run inference
    inference_job = inference(args.test, args.model, uuid, dependencies=generation_job)

    # aggregate, evaluate, report results
    aggregation(args.model, args.method, uuid, dependencies=inference_job)

    print("All jobs queued.")


if __name__ == "__main__":
    os.makedirs(config.experiments, exist_ok=True)
    os.makedirs(os.path.join(config.logs, "pipeline"), exist_ok=True)
    os.makedirs(os.path.join(config.logs, "setup"), exist_ok=True)
    os.makedirs(os.path.join(config.logs, "generation"), exist_ok=True)
    os.makedirs(os.path.join(config.logs, "inference"), exist_ok=True)
    os.makedirs(os.path.join(config.logs, "aggregation"), exist_ok=True)
    parse()
