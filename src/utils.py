import json
import os
from pathlib import Path

def read_jsonl(filepath: str) -> dict:    
    data = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            data.append(example)
    return data


def write_jsonl(data: list, filepath: str) -> None:
    with open(filepath, "w") as f:
        for example in data:
            f.write(json.dumps(example) + "\n")

def append_jsonl(data: list, filepath: str) -> None:
    with open(filepath, "a") as f:
        for example in data:
            f.write(json.dumps(example) + "\n")


def read_txt(filepath: str) -> list:
    data = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            data.append(line.strip())
    return data


def write_txt(data: list, filepath: str) -> None:
    with open(filepath, "w") as f:
        for line in data:
            f.write(line + "\n")


def read_json(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def write_json(d: dict, filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(d, f, indent=4)

def get_experiment_info(experiment_id: int) -> dict():
    project_root = Path(__file__).resolve().parents[1]
    exp_home = os.path.join(project_root, 'experiments')

    exp_summary = os.path.join(exp_home, 'summary.json') 

    exp_info = {}
    # check to see experiment id exists
    if os.path.exists(exp_summary):        
        exp_summary_data = read_json(exp_summary)['summary']

        exp_info = {}
        for exp in exp_summary_data:
            if exp['id'] == experiment_id:
                exp_info = read_json(os.path.join(exp['location'], 'info.json'))
        
        assert exp_info

def format_example(example : dict, dataset: str, includeLabel=False) -> str:
    assert dataset in ['BoolQ']
    
    templates = {
        'BoolQ' : lambda ex: f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"
    }

    text = templates[dataset](example)
    
    if includeLabel:
        text += f"{example['label']}\n"
    
    return text