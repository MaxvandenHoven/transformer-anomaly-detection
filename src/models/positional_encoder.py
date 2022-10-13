import math
import torch
from torch import nn, Tensor


class PositionalEncoder(nn.Module):
    """
    Adapted from https://github.com/KasperGroesLudvigsen/influenza_transformer/blob/main/positional_encoder.py
    """

    def __init__(self, p_dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512):
        """
        Args:
            p_dropout (float, optional): Probability of dropout. Defaults to 0.1.
            max_seq_len (int, optional): Maximum length of the sequence. Defaults to 5000.
            d_model (int, optional): Dimensionality of the mebedding. Defaults to 512.
        """
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.dropout = nn.Dropout(p=p_dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Adds positional embedding to input sequence

        Args:
            x (Tensor): Input sequence

        Returns:
            Tensor: Positionally embedded input sequence
        """
        return self.dropout(x + self.pe[:, : x.size(1)])
