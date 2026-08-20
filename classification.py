import torch
import torch.nn as nn
from typing import Tuple
from denoise import Denoise


# [batch, channel, mels, time]
class EmoClassification(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 1024,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = torch.nn.LSTM(
            512,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.denoise = Denoise()

        self.fc = nn.Linear(hidden_dim, out_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.denoise(x)
        x = self.cnn(x)  # (batch, 512, time)
        x = x.transpose(-1, -2)  # (batch, time, 512)
        x, (_hidden, _cell) = self.lstm(x)  # (batch, time, hidden_dim)
        x = self.fc(x)  # (batch, time, 5)
        x = x.transpose(-1, -2)  # (batch, 5, time)
        x = self.pool(x)  # (batch, 5, 1)
        x = x.squeeze(-1)  # (batch, 5)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    x = torch.randint(0, 100, (3, 80, 224))
    model: EmoClassification = EmoClassification(80, out_channels=4)
    y = model(x.float())
    print(y.shape)
