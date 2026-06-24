from typing import List
import torch
from unet import Unet


# this module will train a random shape data to 4 grade
# flat the n * m data and add embedding
class Classify(torch.nn.Module):
    convs: List[torch.nn.Module]

    def __init__(self, n_mels: int, out_features: int):
        super(Classify, self).__init__()

        self.softmax = torch.nn.Softmax(dim=2)
        self.convs = []
        while n_mels % 2 == 0:
            self.convs.append(
                torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    stride=(2, 2),
                ).cuda()
            )
            n_mels //= 2
        self.linear = torch.nn.Linear(n_mels, out_features)

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x_0 = conv(x_0)
        x_0 = x_0.transpose(-1, -2)
        x_0 = self.linear(x_0)
        x_0, _indices = x_0.max(dim=-2)
        x_0 = self.softmax(x_0)
        x_0 = x_0.squeeze()

        return x_0


if __name__ == "__main__":
    x = torch.rand(3, 1, 80, 224).cuda()

    model = Classify(80, 4).cuda()
    x = model(x)
    print(x)
