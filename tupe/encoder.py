import torch
from torch import nn

from tupe.config import TUPEConfig
from tupe.layers import AbsolutePositionalEmbedding, TUPEEncoderLayer


class TUPEEncoder(nn.Module):
    def __init__(self, config: TUPEConfig) -> None:
        super().__init__()
        pos_embed = AbsolutePositionalEmbedding(config.d_model, config.max_len)
        self.layers = nn.Sequential(
            *[TUPEEncoderLayer(config, pos_embed) for _ in range(config.num_layers)]
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.layers(x)
