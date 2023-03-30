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

def format_example(example : dict, dataset: str, includeLabel=False) -> str:
    superGLUE = ['BoolQ', 'COPA', 'RTE', 'WiC', 'WSC']
    assert dataset in superGLUE

    templates = {
        'BoolQ' : lambda ex: f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:",
        'COPA' : lambda ex: f"Premise: {ex['premise']}\nQuestion: What's the {ex['question']} of this?\nAlternative 1: {ex['choice1']}\nAlternative 2: {ex['choice2']}\nCorrect Alternative:",
        'WSC' : lambda ex: f"Text: {ex['text']}\nQuestion: Does {ex['target']['span2_text']} refer to {ex['target']['span1_text']}?\nAnswer:", 
        'WiC' : lambda ex: f"Context 1: {ex['sentence1']}\nContext 2: {ex['sentence2']}\nWord: {ex['word']}\nSense Match:", 
        'RTE' : lambda ex: f"Text: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nEntailment:"
    }

    text = templates[dataset](example)

    if includeLabel:
        if dataset == 'COPA':
            text += f"{example['label'] + 1}\n" #NOTE: COPA labeling is weird
        else:
            text += f"{example['label']}\n"

    return text

def extract_prediction(output: str, dataset: str):
    superGLUE = ['BoolQ', 'COPA', 'RTE', 'WiC', 'WSC']
    assert dataset in superGLUE

    templates = {
        'BoolQ' : lambda ex: ex.split("\n")[0],
        'COPA' : lambda ex: ex.split("\n")[0],
        'WSC' : lambda ex: ex.split("\n")[0],
        'WiC' : lambda ex: ex.split("\n")[0],
        'RTE' : lambda ex: ex.split("\n")[0],
    }

    return templates[dataset](output)

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

