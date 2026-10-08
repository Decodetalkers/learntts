import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
from .dataset import EmoDataset, EmoDB, EmoBatchCollate
import argparse
import learntts.params as params
from torch.utils.tensorboard import SummaryWriter
from .utils import HUBEDRT_FEATURES
from .attention import SelfAttention

from tqdm.asyncio import tqdm

# Now the features become
# [batch, frames, features]
# ->
# [batch, window_size, features]
# ->
# [batch, 1, features] -> [batch, features]
# ->
# [batch, features_new]


# So just predict there are many features
class HumanPredict(nn.Module):
    def __init__(self, features: int, out_features: int, window_size: int = 100):
        super(HumanPredict, self).__init__()
        self.window_size = window_size
        window_size_next = window_size * 2
        #[batch, window_size, features]
        self.conv_1 = nn.Conv1d(window_size, window_size_next, 3, stride=1, padding=1)
        # [batch, features, window_size]
        self.linear = nn.Linear(window_size_next, 1)
        # [batch, features, 1] -> [batch, features]
        self.linear_end = nn.Linear(features, out_features)

    # this logic will randomly clip the data to slice, and use the slice to gain the identity features
    # They should be generated to the same or similar features
    # TODO: check it later
    def _random_clip(self, data: torch.Tensor) -> torch.Tensor:
        if len(data.shape) == 3:
            return self._random_clip_batch(data)
        frames, features = data.shape

        out = torch.zeros(self.window_size, features)
        if frames < self.window_size:
            slices = self.window_size // frames
            after = self.window_size % frames
            for index in range(slices):
                start = frames * index
                end = start + frames
                out[start:end, :] = data
            start = self.window_size * slices
            out[start:, :] = data[:after, :]
            return out

        end = frames - self.window_size

        import random

        start_point = random.randint(0, end)
        out = data[start_point : start_point + self.window_size, :]
        return out

    # [batch, frames, features]
    def _random_clip_batch(self, data: torch.Tensor) -> torch.Tensor:
        batch, frames, features = data.shape
        out = torch.zeros(batch, self.window_size, features)
        if frames < self.window_size:
            slices = self.window_size // frames
            after = self.window_size % frames
            for index in range(slices):
                start = frames * index
                end = start + frames
                out[:, start:end, :] = data
            start = self.window_size * slices
            out[:, start:, :] = data[:, :after, :]
            return out

        end = frames - self.window_size

        import random

        start_point = random.randint(0, end)
        out = data[:, start_point : start_point + self.window_size, :]
        return out

    # [frames 768] -> [768]
    def forward(self, hubert: torch.Tensor) -> torch.Tensor:
        # [batch ,window_size, features]
        out = self._random_clip(hubert)
        # [batch ,window_features, features]
        out = self.conv_1(out)
        # [batch, features ,window_features]
        out = out.transpose(-1, -2)
        # [batch, features]
        out = self.linear(out).squeeze()
        out = self.linear_end(out).squeeze()
        return out


def main():

    model = HumanPredict(HUBEDRT_FEATURES, 100).to(params.device)
    dataset = EmoDataset(EmoDB, n_fft=params.n_fft, n_mels=params.n_mels)
    batch_collate = EmoBatchCollate(
        dataset.min_div, dataset.emo_features, dataset.mels_count
    )

    train_loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=params.batch_size2,
        collate_fn=batch_collate,
    )
    count = 0
    for _, _, _, hubert in train_loader:
        out = model(hubert.to(params.device))
        print(out.shape)
        count += 1
        if count >= 3:
            break
