
import sys
sys.path.append('/shared/CO/huggingface_/transformers/src')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained("checkpoints-3b/checkpoint-200")
model = AutoModelForCausalLM.from_pretrained("checkpoints-3b/checkpoint-200")
model.to('cuda')

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
        


print('Enter context:', end=' ')
context = input()
print()
print('Enter question:', end=' ')
question = input()
print()
sample = dict(context=context, question=question)
generate_streaming(model, tokenizer, sample)
print()
