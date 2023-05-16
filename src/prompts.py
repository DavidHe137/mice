import sys
sys.path.append("/coc/pskynet6/dhe83/mice/src")
from utils import *
import config

def format_BoolQ(ex: dict)->str:
    return f"{ex['passage']}\nquestion: {ex['question']}\nanswer:"

def format_BoolQ_in_context(ex: dict)->str:
    substitutions = ["no", "yes"]
    return f"{ex['passage']}\nquestion: {ex['question']}\nanswer: {substitutions[ex['label']]}"

def format_general_few_shot(demonstrations, test, dataset):
    context = [ex['in_context'] for ex in demonstrations]

    if dataset in instructions:
        context = [instructions[dataset], *context]

    prompt = format_example(test, dataset)
    prompt = "\n\n".join([*context, prompt])
    return prompt

def format_example(ex : dict, dataset: str):
    assert dataset in config.tasks

    templates = {"BoolQ": format_BoolQ}

    return templates[dataset](ex)

def format_in_context(ex : dict, dataset: str)->str:
    assert dataset in config.tasks

    templates = {"BoolQ": format_BoolQ_in_context}

    return templates[dataset](ex)

def format_few_shot(demonstrations, test, dataset):
    assert dataset in config.tasks

    templates = {"BoolQ": format_general_few_shot}

    return templates[dataset](demonstrations, test, dataset)
