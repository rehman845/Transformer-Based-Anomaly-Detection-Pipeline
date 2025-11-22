from __future__ import annotations

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        latent_dim: int = 64,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_proj = nn.Linear(d_model, input_dim)

        self.latent_head = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.ReLU(),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        encoded = self.encoder(x)
        latent = self.latent_head(encoded)
        latent_mean = latent.mean(dim=1)
        projection = self.projection_head(latent_mean)
        return encoded, latent_mean, projection

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        seq_len = encoded.size(1)
        tgt = torch.zeros_like(encoded)
        tgt = self.positional_encoding(tgt.transpose(0, 1)).transpose(0, 1)
        decoded = self.decoder(tgt, encoded)
        return self.output_proj(decoded)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, latent_mean, projection = self.encode(x)
        reconstruction = self.decode(encoded)
        return reconstruction, latent_mean, projection

