import torch

import numpy as np

class Collater:
    def __init__(self, pad_id):
        self.pad_id = pad_id
    def __call__(self, batch):
        lengths = [len(x['input_ids']) for x in batch]
        max_length = max(lengths)
        
        input_ids = []
        labels = []
        for ix, sample in enumerate(batch):
            input_ids.append([self.pad_id] * (max_length - lengths[ix]) + sample['input_ids'])
            labels.append([self.pad_id] * (max_length - lengths[ix]) + sample['labels'])
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        return dict(input_ids=input_ids, labels=labels)



