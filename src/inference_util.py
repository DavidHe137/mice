import sys
sys.path.append("/coc/pskynet6/dhe83/mice/src")
from utils import *
import config

batch_size = 16
gen_len = 5

def batch_inference(prompts):
    output_tokens = torch.empty(0, dtype=torch.int64).to('cuda:0')

    num_batches = round(len(prompts) / batch_size + 0.5)

    for batch in tqdm(range(num_batches)):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(prompts))

        # tokenize by batch to mitigate effect of long outliers
        tokens = tokenizer(prompts[start:end], padding=True, return_tensors="pt").to('cuda:0')
        outputs = model.generate(
            **tokens,
            max_new_tokens=gen_len,
            temperature=0,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=198,  # special character 'ċ' (bytecode for new line?) NOTE use this for generation
        )
        output_tokens = torch.cat((output_tokens, outputs.sequences[:, -gen_len:]))

    return output_tokens

def batch_scoring(model, prompts):
    log_probs = torch.empty(0, dtype=torch.float32).to('cpu:0')

    ids, prompts = zip(*prompt_map.items())

    num_batches = round(len(prompts) / batch_size + 0.5)
    with torch.no_grad():
        for batch in tqdm(range(num_batches)):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, len(prompts))

            # tokenize by batch to mitigate effect of long outliers
            tokens = tokenizer(prompts[start:end], padding=True, return_tensors="pt").to('cuda:0')

            logits = model(tokens.input_ids, attention_mask=tokens.attention_mask).logits
            labels_attention_mask = tokens.attention_mask.unsqueeze(-1)
            masked_log_probs = labels_attention_mask.float() * torch.log_softmax(
                logits.float(), dim=-1
            )
            seq_token_log_probs = torch.gather(
                masked_log_probs, -1, tokens.input_ids.unsqueeze(-1)
            )
            seq_token_log_probs = seq_token_log_probs.squeeze(dim=-1)
            seq_log_prob = seq_token_log_probs.sum(dim=-1).to("cpu")

            log_probs = torch.cat((log_probs, seq_log_prob))

    return log_probs
