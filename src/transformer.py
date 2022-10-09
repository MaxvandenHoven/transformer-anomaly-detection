from torch import nn, Tensor

from .positional_encoder import PositionalEncoder


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        n_features: int = 1,
        d_model: int = 512,
        nhead: int = 8,
        encoder_dim_feedforward: int = 2048,
        encoder_num_layers: int = 4,
        decoder_dim_feedforward: int = 2048,
        decoder_num_layers: int = 8,
    ):

        super().__init__()

        # Encoder
        self.encoder_input_layer = nn.Linear(
            in_features=n_features, out_features=d_model
        )

        self.positional_encoding_layer = PositionalEncoder(
            p_dropout=0.1, max_seq_len=5000, d_model=d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=encoder_dim_feedforward,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=encoder_num_layers, norm=None
        )

        # Decoder
        self.decoder_input_layer = nn.Linear(
            in_features=n_features, out_features=d_model
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=decoder_dim_feedforward,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=decoder_num_layers, norm=None
        )

        self.linear_mapping = nn.Linear(in_features=d_model, out_features=1)

    def forward(
        self, src: Tensor, tgt: Tensor, src_mask: Tensor = None, tgt_mask: Tensor = None
    ) -> Tensor:
        pass
