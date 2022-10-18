import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from ..utils import read_hdf5, split_windows

class SineTransformerDataset(Dataset):
    def __init__(self, window_size: int = 512, valid: bool = False):
        
        if valid:
            dataset_length = 4000
        else:
            dataset_length = 30000

        grid = torch.stack([torch.linspace(1, window_size, steps=window_size)]*dataset_length)
        freq = ((0.1 - 0.01) * torch.rand((dataset_length)) + 0.01).repeat((window_size, 1)).T
        amplitude = ((1 - 0.05) * torch.rand((dataset_length)) + 0.05).repeat((window_size, 1)).T
        phase = (2 * np.pi * torch.rand((dataset_length))).repeat((window_size, 1)).T

        data = amplitude * torch.sin(freq * grid - phase)


        # Add extra dimension a tthe end for number of features (1)
        data = torch.unsqueeze(data, -1)

        # Convert to tensor
        self.data = Tensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_sine_dataloaders(window_size: int = 512, batch_size: int = 128,
        shuffle: bool = True):

    train_dataset = SineTransformerDataset(window_size, valid=False)
    valid_dataset = SineTransformerDataset(window_size, valid=True)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    )