import torch
import torch.nn.functional as F

from copy import deepcopy
from torch import nn, Tensor

from .positional_encoder import PositionalEncoder


class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048, 
            dropout: float = 0.1, layer_norm_eps: float = 1e-5, batch_first: bool = True, 
            device=None, dtype=None) -> None:
        """
        Args:
            d_model (int, optional): number of expected features in the input. 
                Defaults to 512.
            nhead (int, optional): number of heads in the multiheadattention models. 
                Defaults to 8.
            dim_feedforward (int, optional): dimension of the feedforward network model. 
                Defaults to 2048.
            dropout (float, optional): Dropout value. Defaults to 0.1.
            layer_norm_eps (float, optional): the eps value in layer normalization 
                components. Defaults to 1e-5.
            batch_first (bool, optional): Defaults to True.
            device (_type_, optional): Defaults to None.
            dtype (_type_, optional): Defaults to None.
        """        
        factory_kwargs = {"device": device, "dtype": dtype}
        super(CustomEncoderLayer, self).__init__()

        # Self attention module
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )

        # Feedforward module
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """ 
        Args:
            src (Tensor): sequence to the encoder of shape (N, L, E).
            src_mask (Tensor, optional): mask for src sequence of shape (L, L). 
                Defaults to None.

        Returns:
            output: encoder layer output of shape (N, L, E).
            attn_output_weights: attention weights per head of shape (N, nhead, L, L).
        """        
        x = src
        
        # Self attention block
        attn_output, attn_output_weights = self.self_attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            attn_mask=src_mask,
            key_padding_mask=None,
            need_weights=True,
            average_attn_weights=False,
        )
        attn_output = self.dropout1(attn_output)

        x = x + attn_output

        # Feedforward block
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(self.norm2(x)))))
        ff_output = self.dropout2(ff_output)

        x = x + ff_output

        return x, attn_output_weights


class CustomEncoder(nn.Module):
    def __init__(self, encoder_layer: CustomEncoderLayer, num_layers: int) -> None:
        """
        Args:
            encoder_layer (CustomEncoderLayer): encoder layer to replicate.
            num_layers (int): num of encoder layers.
        """        
        super(CustomEncoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Args:
            src (Tensor): sequence to the encoder of shape (N, L, E).
            src_mask (Tensor, optional): mask for src sequence of shape (L, L). 
                Defaults to None.

        Returns:
            output: output of all encoder layers of shape (N, L, E).
            layer_attn_output_weights: attention weights per head per layer of shape
                (N, num_layers, nhead, L, L).
        """        
        output = src
        layer_attn_output_weights = []

        for layer in self.layers:
            output, attn_output_weights = layer(output, src_mask=src_mask)
            layer_attn_output_weights.append(attn_output_weights)

        layer_attn_output_weights = torch.stack(layer_attn_output_weights, dim=1)

        return output, layer_attn_output_weights

        
class CustomTransformer(nn.Module):
    def __init__(self, n_features: int = 1, max_seq_len: int = 128, pe_dropout: float = 0.1, 
            num_layers: int = 4, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048, 
            dropout: float = 0.1, layer_norm_eps: float = 1e-5) -> None:
        """_summary_

        Args:
            n_features (int, optional): _description_. Defaults to 1.
            max_seq_len (int, optional): _description_. Defaults to 128.
            pe_dropout (float, optional): _description_. Defaults to 0.1.
            num_layers (int, optional): _description_. Defaults to 4.
            d_model (int, optional): _description_. Defaults to 512.
            nhead (int, optional): _description_. Defaults to 8.
            dim_feedforward (int, optional): _description_. Defaults to 2048.
            dropout (float, optional): _description_. Defaults to 0.1.
            layer_norm_eps (float, optional): _description_. Defaults to 1e-5.
        """ 
        super(CustomTransformer, self).__init__()    

        self.input_embedding = nn.Linear(in_features=n_features, out_features=d_model)

        self.positional_encoder = PositionalEncoder(
            p_dropout=pe_dropout,
            max_seq_len=max_seq_len,
            d_model=d_model    
        )

        self.transformer_encoder_layer = CustomEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )

        self.transformer_encoder = CustomEncoder(
            encoder_layer=self.transformer_encoder_layer,
            num_layers=num_layers
        )

        self.linear_mapping = nn.Linear(in_features=d_model, out_features=n_features)

        self.activation = nn.Tanh()

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Args:
            src (Tensor): sequence to the encoder of shape (N, L, E).
            src_mask (Tensor, optional): mask for src sequence of shape (L, L). 
                Defaults to None.

        Returns:
            output: output of all encoder layers of shape (N, L, E).
            layer_attn_output_weights: attention weights per head per layer of shape
                (N, num_layers, nhead, L, L).
        """ 

        src = self.positional_encoder(self.input_embedding(src))

        output, layer_attn_output_weights = self.transformer_encoder(src, src_mask)

        output = self.activation(self.linear_mapping(output))

        return output, layer_attn_output_weights
