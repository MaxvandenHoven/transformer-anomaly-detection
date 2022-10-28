import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from ..utils import read_hdf5, split_windows

class HelicopterTransformerDataset(Dataset):
    def __init__(self, window_size: int = 512, valid: bool = False):
        # Checkk that window size divides total sequence length for simplicity
        assert 61440 % window_size == 0, "Please select dividor of 61440 for window size."

        # Load data from h5 file
        if valid:
            data = read_hdf5("data\helicopter\dfvalid.h5", "dfvalid")
        else:
            data = read_hdf5("data\helicopter\dftrain.h5", "dftrain")
        
        # Scaling (min max every sample since "All data has been multiplied by a factor 
        # so that absolute values are meaningless")
        nominator = data - data.min(axis=1).reshape(-1, 1)
        denominator = (data.max(axis=1) - data.min(axis=1) + 1e-5).reshape(-1, 1)
        data = nominator/denominator * 2 - 1

        # Add extra dimension a tthe end for number of features (1)
        data = np.expand_dims(split_windows(data), -1)

        # Convert to tensor
        self.data = Tensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_helicopter_dataloaders(window_size: int = 512, batch_size: int = 128,
        shuffle: bool = True):

    train_dataset = HelicopterTransformerDataset(window_size, valid=False)
    valid_dataset = HelicopterTransformerDataset(window_size, valid=True)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    )


class HelicopterAutoencoderDataset(Dataset):
    def __init__(self):
        data = read_hdf5("data\helicopter\dftrain.h5", "dftrain")

        nominator = data - data.min(axis=1).reshape(-1, 1)
        denominator = (data.max(axis=1) - data.min(axis=1) + 1e-5).reshape(-1, 1)
        data = nominator/denominator * 2 - 1

        data = np.expand_dims(data, -1)

        # Convert to tensor
        self.data = Tensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
