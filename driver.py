
#%%

from tokenizer import Tokenizer
from datasets import load_from_disk
from transformers import LlamaConfig, LlamaForCausalLM
from utils import collater
from torch.utils.data import DataLoader



tokenizer = Tokenizer('data/tok3072.model')
dataset  = load_from_disk('data/preprocessed_dataset')
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collater)



config = LlamaConfig()

config.vocab_size = tokenizer.n_words
config.hidden_size = 512
config.intermediate_size = 512
config.num_hidden_layers = 6
config.num_attention_heads = 8
config.max_position_embeddings = 512
config.pad_token_id = tokenizer.pad_id
config.bos_token_id = tokenizer.bos_id
config.eos_token_id = tokenizer.eos_id

model = LlamaForCausalLM._from_config(config)



# %%
