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

def get_experiment_info(experiment_id: str) -> dict():
    project_root = Path(__file__).resolve().parents[1]
    exp_home = os.path.join(project_root, 'experiments')

    exp_summary = os.path.join(exp_home, 'summary.json')

    exp_info = {}
    if os.path.exists(exp_summary):
        exp_summary_data = read_json(exp_summary)
        assert experiment_id in exp_summary_data

        exp_info = read_json(os.path.join(exp_summary_data[experiment_id]['location'], 'info.json'))
        assert exp_info

    return exp_info

def format_example(example : dict, dataset: str, includeLabel=False) -> str:
    assert dataset in ['BoolQ']

    templates = {
        'BoolQ' : lambda ex: f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"
    }

    text = templates[dataset](example)

    if includeLabel:
        text += f"{example['label']}\n"

    return text

def extract_prediction(output: str, dataset: str):
    assert dataset in ['BoolQ']

    templates = {
        'BoolQ' : lambda ex: ex.split("\n")[0]
    }

    return templates[dataset](output)

def verbalize(pred: str, dataset: str):
    superGLUE = ['BoolQ', 'CB', 'COPA', 'MultiRC', 'ReCoRD', 'RTE', 'WiC', 'WSC']
    assert dataset in superGLUE
    
    templates = {
        'BoolQ' : lambda pred: pred.lower() in ["yes", "true"]
    }

    return templates[dataset](pred)

def get_log_with_uuid(uuid: str) -> dict():
    project_root = Path(__file__).resolve().parents[1]
    log_file = os.path.join(project_root, 'logs', f"{uuid}.json")
    return read_json(log_file)