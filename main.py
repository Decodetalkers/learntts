import torch
from unet import Unet
# this module will train a random shape data to 4 grade
# flat the n * m data and add embedding
class Classify(torch.nn.Module):
    def __init__(self, features: int, hidden_dim: int):
        super(Classify, self).__init__()
        self.embedding = torch.nn.Embedding(features, hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, 4)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        x_0 = self.embedding(x_0)
        x_0 = self.linear(x_0)
        x_0 = x_0.argmax(dim=1)
        x_0 = x_0.argmax(dim=1)
        x_0 = self.softmax(x_0.float())
        return x_0

if __name__ == "__main__":
    x = torch.randint(0, 100, (20, 100, 30))
    print(x.shape)
    model = Classify(100, 150)
    x = model(x)
    print(x.shape)
    print(x)
