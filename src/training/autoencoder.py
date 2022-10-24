import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from ..models.transformer import CustomTransformer
from ..models.autoencoder import AutoEncoderGarcia2020


def train_autoencoder_garcia2020(
    train_data: DataLoader,
    valid_data: DataLoader,
    transformer_model: CustomTransformer,
    autoencoder_model: AutoEncoderGarcia2020,
    optimizer: optim.Optimizer,
    
)