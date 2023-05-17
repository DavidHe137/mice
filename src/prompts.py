import sys
sys.path.append("/coc/pskynet6/dhe83/mice/src")
import config
import re

def format_BoolQ(ex: dict)->str:
    return f"{ex['passage']}\nquestion: {ex['question']}\nanswer:"

def format_BoolQ_in_context(ex: dict)->str:
    substitutions = ["no", "yes"]
    return " ".join([format_BoolQ(ex), substitutions[ex['label']]])

def format_CB(ex: dict)->str:
    return f"{ex['premise']}\nquestion: {ex['hypothesis']}. true, false, or neither?\nanswer:"

def format_CB_in_context(ex: dict)->str:
    substitutions = {"contradiction": "false", "entailment": "true", "neutral": "neither"}
    return " ".join([format_CB(ex), substitutions[ex['label']]])

def format_RTE(ex:dict)->str:
    return (f"{ex['premise']}"
            f"\nquestion: {ex['hypothesis']} True or False?"
            f"\nanswer:")

def format_RTE_in_context(ex:dict)->str:
    substitutions = {"entailment": "True", "not_entailment": "False"}
    return " ".join([format_RTE(ex), substitutions[ex['label']]])

def format_WiC(ex:dict)->str:
    return (f"{ex['sentence1']}\n{ex['sentence2']}\n"
            f"question: Is the word \'{ex['word']}\' used in the same way in the two sentences above?"
            f"answer:")

def format_WiC_in_context(ex:dict)->str:
    substitutions = ["no", "yes"]
    return " ".join([format_WiC(ex), substitutions[ex['label']]])

def format_WSC(ex: dict)->str:
    return (f"Passage: {ex['text']}\n"
            f"Question: In the passage above, does the pronoun "
            f"\"{ex['target']['span2_text']}\" refer to {ex['target']['span1_text']}?\n"
            f"Answer:")

def format_WSC_in_context(ex: dict)->str:
    substitutions = ["no", "yes"]
    return " ".join([format_WSC(ex), substitutions[ex['label']]])

def format_Winograd(ex:dict)->str:
    text = ex['text'].split(" ")
    pronoun = ex['target']['span2_text']
    index = ex['target']['span2_index']
    assert pronoun == text[index]
    text[index] = "".join(["*", text[index], "*"])

    passage = " ".join(text)
    return (f"Passage: {passage}"
             f"\nQuestion: In the passage above, what does the pronoun \"*{pronoun}*\" refer to?\nAnswer:")

def format_Winograd_in_context(ex:dict)->str:
    return " ".join([format_WSC(ex), ex['target']['span1_text']])

def format_general_few_shot(demonstrations, test, dataset):
    context = [ex['in_context'] for ex in demonstrations]

    instructions = {"Winograd": "Final Exam with Answer Key\nInstructions: Please carefully read the following passages. For each passage, you must identify which noun the pronoun marked in *bold* refers to.\n====="}
    if dataset in instructions:
        context = [instructions[dataset], *context]

    prompt = format_example(test, dataset)
    prompt = "\n\n".join([*context, prompt])
    return prompt

def format_example(ex : dict, dataset: str):
    assert dataset in config.tasks

    templates = {"BoolQ": format_BoolQ,
                "CB": format_CB,
                "RTE": format_RTE,
                "WiC": format_WiC,
                "WSC": format_WSC,
                "Winograd": format_Winograd}

    return templates[dataset](ex)

def format_in_context(ex : dict, dataset: str)->str:
    assert dataset in config.tasks

    templates = {"BoolQ": format_BoolQ_in_context,
                "CB": format_CB_in_context,
                "RTE": format_RTE_in_context,
                "WiC": format_WiC_in_context,
                "WSC": format_WSC_in_context,
                "Winograd": format_Winograd_in_context}

    return templates[dataset](ex)

def format_few_shot(demonstrations, test, dataset):
    assert dataset in config.tasks

    templates = {"BoolQ": format_general_few_shot,
                "CB": format_general_few_shot,
                "RTE": format_general_few_shot,
                "WiC": format_general_few_shot,
                "WSC": format_general_few_shot,
                "Winograd": format_general_few_shot}

    return templates[dataset](demonstrations, test, dataset)

def first_word(s):
        return "".join([c for c in re.split(" |\n|</s>",s.strip())[0] if str.isalpha(c)]).lower()

def verbalize_TrueFalse(text):
    if first_word(text) in ["yes", "true"]:
        return True
    elif first_word(text) in ["no", "false"]:
        return False
    else:
        return None

def verbalize_CB(text):
    if first_word(text) in ["yes", "true"]:
        return "entailment"
    elif first_word(text) in ["no", "false"]:
        return "contradiction"
    elif first_word(text) in ["neither", "neutral"]:
        return "neutral"
    else:
        return None

def verbalize_RTE(text):
    if first_word(text) in ["yes", "true"]:
        return "entailment"
    elif first_word(text) in ["no", "false"]:
        return "not_entailment"
    else:
        return None

def common_words(s1, s2):
    s1, s2 = s1.split(" "), s2.split(" ")
    return len(set(s1).intersection(set(s2)))

def verbalize_Winograd(text):
    first_sentence = text.split("\n")[0].lower().rstrip('.')
    return re.sub("(.+) refers to ", "", first_sentence).strip()

def verbalize(text : dict, dataset: str):
    assert dataset in config.tasks

    templates = {"BoolQ": verbalize_TrueFalse,
                 "CB": verbalize_CB,
                "RTE": verbalize_RTE,
                "WiC": verbalize_TrueFalse,
                "WSC": verbalize_TrueFalse,
                "Winograd": verbalize_Winograd}

    return templates[dataset](text)

def validate(pred, label, dataset):
    assert dataset in config.tasks

    equality = lambda pred, label: pred == label
    winograd = lambda pred, label: label in pred or label in pred or common_words(pred, label) >= len(label.split(" ")) / 2
    templates = {"BoolQ": equality,
             "CB": equality,
            "RTE": equality,
            "WiC": equality,
            "WSC": equality,
            "Winograd": winograd}

    return templates[dataset](pred, label)

