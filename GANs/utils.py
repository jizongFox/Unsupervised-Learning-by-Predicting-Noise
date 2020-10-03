from collections import Iterator

import torch
from torch.utils.data.sampler import Sampler


class _InfiniteRandomIterator(Iterator):
    def __init__(self, data_source, shuffle=True):
        self.data_source = data_source
        self.shuffle = shuffle
        if self.shuffle:
            self.iterator = iter(torch.randperm(len(self.data_source)).tolist())
        else:
            self.iterator = iter(
                torch.arange(start=0, end=len(self.data_source)).tolist()
            )

    def __next__(self):
        try:
            idx = next(self.iterator)
        except StopIteration:
            if self.shuffle:
                self.iterator = iter(torch.randperm(len(self.data_source)).tolist())
            else:
                self.iterator = iter(
                    torch.arange(start=0, end=len(self.data_source)).tolist()
                )
            idx = next(self.iterator)
        return idx


class InfiniteRandomSampler(Sampler):
    def __init__(self, data_source, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        return _InfiniteRandomIterator(self.data_source, shuffle=self.shuffle)

    def __len__(self):
        return len(self.data_source)
