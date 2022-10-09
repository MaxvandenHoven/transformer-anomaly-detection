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
