import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, channels: int = 0, max_len=5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim)
        )

        if channels == 0:
            pe = torch.zeros(max_len, dim)

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(channels, max_len, dim)

            pe[:, :, 0::2] = torch.sin(position * div_term)
            pe[:, :, 1::2] = torch.cos(position * div_term)

        self.channels = channels
        # [1, channels, max_len, dim]
        # or [1, max_len, dim]
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        if self.channels == 0:
            # x: [batch, seq_len, dim]
            return x + self.pe[:, : x.size(1)]  # ty: ignore[not-subscriptable]
        # x: [batch, channels,seq_len, dim]
        return x + self.pe[:, :, : x.size(2)]  # ty: ignore[not-subscriptable]


class SelfAttention(nn.Module):
    def __init__(self, dim: int, channels: int = 0):
        super().__init__()

        self.positional_encoding = PositionalEncoding(dim, channels)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        # [batch, channels,seq_len, dim]
        x = self.positional_encoding(x)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (Q.size(-1) ** 0.5)

        attention = torch.softmax(scores, dim=-1)

        return attention @ V
