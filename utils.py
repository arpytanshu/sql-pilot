import torch

def collater(batch, pad_id=-1):
    lengths = [len(x) for x in batch['input_ids']]
    max_length = max(lengths)
    for i in range(len(batch['input_ids'])):
        batch['input_ids'][i] += [pad_id] * (max_length - lengths[i])
        batch['labels'][i] += [pad_id] * (max_length - lengths[i])

    input_ids = torch.tensor(batch['input_ids'])
    labels = torch.tensor(batch['labels'])

    return dict(input_ids=input_ids, labels=labels)