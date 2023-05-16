import json
import os
import config

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

def get_dir_with_id(parent, idx):
    found= False
    for d in os.listdir(parent):
        if os.path.isdir(os.path.join(parent,d)):
            i = int(d.split(config.delim)[0])
            if i == idx:
                found = True
                child = os.path.join(parent,d)

    assert found
    return child

def new_dir_id(parent):
    idx = 1
    for d in os.listdir(parent):
        if os.path.isdir(os.path.join(parent,d)):
            existing_idx = int(d.split(config.delim)[0])
            idx = max(idx, existing_idx + 1)
    return idx

def get_experiment_info(experiment_id: str) -> dict():
    exp_summary = os.path.join(config.experiments, 'summary.json')

    exp_info = {}
    if os.path.exists(exp_summary):
        exp_summary_data = read_json(exp_summary)
        assert experiment_id in exp_summary_data

        exp_info = read_json(os.path.join(exp_summary_data[experiment_id]['location'], 'info.json'))
        assert exp_info

    return exp_info

def get_log_with_uuid(uuid: str) -> dict():
    log_file = os.path.join(config.logs, f"{uuid}.json")
    return read_json(log_file)

def verbalize(pred: str, dataset: str):
    superGLUE = ['BoolQ', 'COPA', 'RTE', 'WiC', 'WSC']
    assert dataset in superGLUE

    def isInt(pred: str) -> bool:
        if pred is None:
            return False
        try:
            int(pred)
            return True
        except ValueError:
            return False


    templates = {
        'BoolQ' : lambda pred: pred.lower() in ["yes", "true"],
        'COPA' : lambda pred: int(pred) - 1 if isInt(pred) else "",
        'WSC' : lambda pred: pred.lower() in ["yes", "true"],
        'WiC' : lambda pred: pred.lower() in ["yes", "true"],
        'RTE' : lambda pred: pred.lower() in ["entailment"],
    }

    return templates[dataset](pred)

def evaluate(pred: list, labels: list, dataset: str) -> float:
    def accuracy(pred: list, labels: list):
        correct = 0
        total = 0
    return 0.0

