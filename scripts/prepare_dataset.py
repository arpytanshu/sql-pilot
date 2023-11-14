
import os
import fire
import torch
from datasets import load_dataset, load_from_disk
from tokenizer import Tokenizer


get_ctx = lambda x: f"context : {x['context']} \nquestion : {x['question']}"
get_ans = lambda x: f"answer : {x['answer']}"

def preprocess(sample):

    ctx = get_ctx(sample)
    ans = get_ans(sample)

    ctx_ids = tokenizer.encode(ctx, bos=True, eos=False)
    ans_ids = tokenizer.encode(ans, bos=False, eos=True)

    labels = [-100] * (len(ctx_ids)-1) + ans_ids + [-100]
    inputs = ctx_ids + ans_ids

    return {'input_ids': inputs, 'labels': labels}

def run(tokenizer_path, data_cache_dir="data"):
    cache_path = os.path.join(data_cache_dir, 'preprocessed_dataset')
    if not os.path.exists(cache_path):
        global tokenizer
        dataset = load_dataset("b-mc2/sql-create-context", verification_mode='no_checks')
        tokenizer = Tokenizer(tokenizer_path)
        preprocessed_dataset = dataset.map(preprocess)
        preprocessed_dataset.save_to_disk(cache_path)
        print(f"dataset saved at {cache_path}")
    else:
        preprocessed_dataset = load_from_disk(cache_path)
        print(f"dataset already present at {cache_path}")

if __name__ == '__main__':
    fire.Fire(run)
