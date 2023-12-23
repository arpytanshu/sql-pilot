
import sys
import warnings

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import format

warnings.filterwarnings("ignore")


def generate_streaming(model, tokenizer, sample, do_sample = False, temperature=1.0, max_length=100):
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
        
        input_ids_npy = input_ids.ravel()[input_len:].cpu().numpy()
        string = tokenizer.decode(input_ids_npy, skip_special_tokens=True)
        sys.stdout.write("\rSQL: {} ".format(string))


def main(checkpoint_path='checkpoints/checkpoints-3b/checkpoint-200'):
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Start the conversation
    print("Type 'exit' to quit script.")
    print("Type 'new' to provide new context.")
    print("-----------------------------------")

    while True:
        print('\n\n\nEnter context:', end=' ')
        context = input()

        if context.lower().strip() == 'exit':
            sys.exit(0)
        
        while True:
            question = input('\nEnter question:')
            if question.lower().strip() == 'exit':
                sys.exit(0)
            if question.lower().strip() == 'new':
                break
            sample = dict(context=context, question=question)
            generate_streaming(model, tokenizer, sample)
            print()
        
        
if __name__ == '__main__':
    fire.Fire(main)

# CREATE TABLE table_name(name VARCHAR, age INTEGER, gender VARCHAR, preference VARCHAR)
# get me first names of all people born after 2000 who prefer read_meat.
 