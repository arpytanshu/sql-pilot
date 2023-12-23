
import os
import bitsandbytes as bnb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from utils import get_datasets, Collater, generate, custom_evaluate
from transformers import TrainerCallback


# model_id = "meta-llama/Llama-2-7b-chat-hf"
# model_id = "openlm-research/open_llama_3b_v2"
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
dataset_name = "data/preprocessed_dataset"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # load_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

datasets = get_datasets(tokenizer, dataset_name, masked_labels=True)
train_dataset = datasets['train_dataset']
test_dataset = datasets['test_dataset']


class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"
    def __init__(self, eval_dataset):
        super().__init__()
        self.eval_dataset = eval_dataset

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        print("sample generation...")
        ix = np.random.randint(0, len(self.eval_dataset))
        sample = self.eval_dataset[ix]
        res = generate(model, tokenizer, sample)
        print('ground_truth:: ', res['ground_truth'].strip())
        print('  generation:: ', res['generation'].strip())
    
        print("running custom_evaluate...")
        num_samples = 100
        exact_match = custom_evaluate(model, tokenizer, self.eval_dataset, num_samples=num_samples)
        print(f"{exact_match} exact match out of {num_samples} samples.")


training_args = transformers.TrainingArguments(
    # auto_find_batch_size=True,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=48,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=1,
    logging_steps=25,
    eval_steps=100,
    output_dir='checkpoints/tiny_llama-masked_labels/',
    save_strategy='steps',
    save_steps=50,
    resume_from_checkpoint=True,
)

trainer = transformers.Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=training_args,
    data_collator=Collater(pad_id=tokenizer.pad_token_id),
    callbacks=[MyCallback(test_dataset)]
)

model.config.use_cache = False

# trainer.evaluate()
trainer.train()
# trainer.train(resume_from_checkpoint=True)

