from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset, Sampler


class PairedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gcms_vec, smell_vec = self.data[idx]
        if not isinstance(gcms_vec, torch.Tensor):
            gcms_vec = torch.tensor(gcms_vec, dtype=torch.float32)
        else:
            gcms_vec = gcms_vec.detach().clone()
        if not isinstance(smell_vec, torch.Tensor):
            smell_vec = torch.tensor(smell_vec, dtype=torch.float32)
        else:
            smell_vec = smell_vec.detach().clone()
        return gcms_vec, smell_vec


class UniqueGCMSampler(Sampler):
    def __init__(self, data, batch_size: int):
        self.data = data
        self.batch_size = batch_size
        self.gcms_to_indices = {}
        for idx, (gcms_vec, _) in enumerate(data):
            self.gcms_to_indices.setdefault(tuple(gcms_vec), []).append(idx)
        for idx_list in self.gcms_to_indices.values():
            random.shuffle(idx_list)
        self.unique_gcms = list(self.gcms_to_indices.keys())

    def __iter__(self):
        gcms_queues = {key: list(indices) for key, indices in self.gcms_to_indices.items()}
        all_batches = []
        while any(gcms_queues.values()):
            random.shuffle(self.unique_gcms)
            current_batch = []
            for gcms_key in self.unique_gcms:
                queue = gcms_queues[gcms_key]
                if queue:
                    current_batch.append(queue.pop())
                    if len(current_batch) == self.batch_size:
                        all_batches.append(current_batch)
                        current_batch = []
            if current_batch:
                all_batches.append(current_batch)
        return iter([idx for batch in all_batches for idx in batch])

    def __len__(self):
        return len(self.data)
