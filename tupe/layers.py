import math

import torch
from torch import nn
from torch.nn import functional as F

from tupe.attention import TUPEMultiHeadAttention
from tupe.config import TUPEConfig


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.empty(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int) -> torch.tensor:
        # shape (1, seq_len, d_model)
        return self.pe[:, :seq_len]


class FeedForward(nn.Module):
    def __init__(self, config: TUPEConfig) -> None:
        super().__init__()
        d_hidden = config.d_model * config.expansion_factor
        self.fc1 = nn.Linear(config.d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.gelu(self.fc1(x))
        out = self.dropout(self.fc2(x))
        return out


class TUPEEncoderLayer(nn.Module):
    def __init__(
        self, config: TUPEConfig, pos_embed: AbsolutePositionalEmbedding
    ) -> None:
        super().__init__()
        self.ffn = FeedForward(config)
        self.attn = TUPEMultiHeadAttention(config, pos_embed)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.attn_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
