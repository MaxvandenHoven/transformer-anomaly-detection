import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from ..utils import read_hdf5, split_windows

class HelicopterTransformerTrainDataset(Dataset):
    def __init__(self, window_size: int = 512, device: str = "cuda"):
        # Checkk that window size divides total sequence length for simplicity
        assert 61440 % window_size == 0, "Please select dividor of 61440 for window size."

        # Load data from h5 file
        self.data = read_hdf5("data\helicopter\dftrain.h5", "dftrain")

        # Add extra dimension a tthe end for number of features (1)
        self.data = np.expand_dims(split_windows(self.data), -1)

        # Scaling
        # TODO

        # Convert to tensor
        self.data = Tensor(self.data).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]


def HelicopterTransformerTrainDataloader(window_size: int=512, batch_size: int = 128, 
        shuffle: bool = True, device: str = "cuda"):
    dataset = HelicopterTransformerTrainDataset(window_size=window_size, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)