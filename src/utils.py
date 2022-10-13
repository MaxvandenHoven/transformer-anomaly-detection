# Imports
import h5py
import numpy as np


def read_hdf5(filename: str, outer_name: str) -> np.array:
    """Extracts dataset from HDF5 file

    Args:
        filename (str): Pathlike to HDF5 file
        outer_name (str): Name of group containing dataset values

    Returns:
        np.array: data
    """
    with h5py.File(filename, "r") as f:
        data = np.array(f[outer_name]["block0_values"])
        return data


def split_windows(data: np.array, window_size: int = 512) -> np.array:
    """_summary_

    Args:
        data (np.array): Data of shape (n_batches, n_steps)
        window_size (int, optional): Sliding window size. Assumed to be a dividor of
            `n_steps`. Defaults to 512.

    Returns:
        np.array: Split data of shape (n_batches * (n_steps / window_size), window_size)
    """
    assert len(data.shape) == 2, "Dataset is not 2-dimensional"
    n_steps = data.shape[1]

    assert n_steps % window_size == 0, "window_size does not divide n_steps"
    n_windows = int(n_steps / window_size)

    return np.concatenate([np.reshape(row, (n_windows, window_size)) for row in data])
