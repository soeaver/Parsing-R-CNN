import torch
from torch.utils.data.sampler import Sampler


class RangeSampler(Sampler):
    def __init__(self, start_ind, end_ind):
        self.start_ind = start_ind
        self.end_ind = end_ind

    def __iter__(self):
        indices = torch.arange(self.start_ind, self.end_ind).tolist()
        return iter(indices)

    def __len__(self):
        return self.end_ind - self.start_ind
