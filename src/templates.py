def format_example(example : dict, dataset: str, includeLabel=False) -> str:
    assert dataset in ['BoolQ']
    
    templates = {
        'BoolQ' : lambda ex: f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"
    }

    text = templates[dataset](example)
    
    if includeLabel:
        text += f"{example['label']}\n"
    
    return text
    
