
import torch
import numpy as np
from datasets import load_from_disk

def format(sample):
    inst_str = 'Create appropriate SQL for this question.'
    ctx = f"{inst_str} context : {sample['context']} \nquestion : {sample['question']} \nanswer: "
    if sample.get('answer'):
        ans = f"{sample['answer']}"
        return ctx, ans
    else:
        return ctx, None

def get_datasets(tokenizer, dataset_path, masked_labels=True):
    
    def _masked_labels_preprocess(sample):
        ctx, ans = format(sample)
        ctx_ids = [tokenizer.bos_token_id] + tokenizer.encode(ctx, add_special_tokens=False)
        ans_ids = tokenizer.encode(ans, add_special_tokens=False) + [tokenizer.eos_token_id]
        inputs = ctx_ids + ans_ids
        labels = [-100] * (len(ctx_ids)) + ans_ids
        return {'input_ids': inputs, 'labels': labels}

    def _unmasked_labels_preprocess(sample):
        ctx, ans = format(sample)
        ctx_ids = [tokenizer.bos_token_id] + tokenizer.encode(ctx, add_special_tokens=False)
        ans_ids = tokenizer.encode(ans, add_special_tokens=False) + [tokenizer.eos_token_id]
        inputs = ctx_ids + ans_ids
        labels = ctx_ids + ans_ids
        return {'input_ids': inputs, 'labels': labels}

    preprocess_fn = _masked_labels_preprocess if masked_labels else _unmasked_labels_preprocess

    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(preprocess_fn)

    rng = np.random.default_rng(seed=2310)
    test_indices = rng.integers(0, len(dataset), int(0.05 * len(dataset)))
    train_indices = np.setdiff1d(np.arange(len(dataset)), test_indices) 

    test_dataset = dataset.select(test_indices)
    train_dataset = dataset.select(train_indices)

    return dict(train_dataset=train_dataset, test_dataset=test_dataset)


class Collater:
    def __init__(self, pad_id):
        self.pad_id = pad_id
    def __call__(self, batch):
        lengths = [len(x['input_ids']) for x in batch]
        max_length = max(lengths)
        
        input_ids = []
        labels = []
        attention_mask = []
        for ix, sample in enumerate(batch):
            input_ids.append([self.pad_id] * (max_length - lengths[ix]) + sample['input_ids'])
            labels.append([-100] * (max_length - lengths[ix]) + sample['labels'])
            attention_mask.append([0] * (max_length - lengths[ix]) + [1] * len(sample['input_ids']))
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        attention_mask = torch.tensor(attention_mask)
        return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def generate(model, tokenizer, sample, do_sample = False, temperature=1.0, max_length=100):

    ctx, ans = format(sample)
    ctx_ids = [tokenizer.bos_token_id] + tokenizer.encode(ctx, add_special_tokens=False)
    input_ids = torch.tensor(ctx_ids).unsqueeze(0).to(model.device)
    
    model.eval()
    input_len = len(input_ids[0])
    for ix in range(max_length):
        with torch.no_grad():
            output = model(input_ids.to(model.device), 
                            output_hidden_states=True, 
                            use_cache=False)
            logits = output.logits[:, -1, :] # BxT
        logits = logits / temperature
        probs = torch.softmax(logits, axis=1) # BxT
        if do_sample:
            out_token_id = torch.multinomial(probs, 1) # Bx1
        else:
            out_token_id = torch.argmax(probs, axis=1).unsqueeze(1)
        input_ids = torch.cat([input_ids, out_token_id], dim=-1)
        
        if out_token_id.item() == tokenizer.eos_token_id:
            break

    generation_ids = input_ids[:, input_len-1:]
    generation = tokenizer.decode(generation_ids.view(-1).tolist(), skip_special_tokens=True)

    res = {}
    if ans:
        res['ground_truth'] = ans
    res['generation'] = generation  

    return res



def custom_evaluate(model, tokenizer, dataset, num_samples):
    generations = []
    ground_truths = []
    for _ in range(num_samples):
        print('|', end='')
        sample_ix = np.random.randint(0, len(dataset))
        sample = dataset[sample_ix]
        res = generate(model, tokenizer, sample, do_sample=False, temperature=1.0, max_length=100)
        generations.append(res['generation'])
        ground_truths.append(res['ground_truth'])

    exact_match = 0
    for gen, gt in zip(generations, ground_truths):
        gen = gen.strip()
        gt = gt.strip()
        if gen == gt:
            exact_match += 1
        elif gen.replace('\"', '') == gt.replace('\"', ''):
            exact_match += 1
    return exact_match
    
    