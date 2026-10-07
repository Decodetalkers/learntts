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

# So just predict there are many features 
class HumanPredict(nn.Module):
    def __init__(self, features: int, out_features: int):
        super(HumanPredict, self).__init__()
        features_2 = features
        self.conv_1 = nn.Conv1d(features_2, features_2 // 2, 3, stride=2, padding=1)
        features_2 //= 2
        self.conv_2 = nn.Conv1d(features_2, features_2 // 2, 3, stride=2, padding=1)
        self.features = features_2 // 2
        self.linear = nn.Linear(self.features, out_features)
        self.pool = nn.AdaptiveMaxPool1d(1)
    # [frames 768] -> [768]
    def forward(self, hubert: torch.Tensor) -> torch.Tensor:
        # [768, frames]
        out = hubert
        if len(out.shape) != 3:
            out = out.unsqueeze(-3)
        # [1, 768, frames]
        out = out.transpose(-1, -2)
        out = self.conv_1(out)
        out = self.conv_2(out)
        # [features_new, frames_??]
        out = out.squeeze()
        # [frames_??, features_new]
        out = out.transpose(-1, -2)
        # [frames_??, out_features]
        out = self.linear(out)

        # [out_features, frames_??]
        out = out.transpose(-1, -2)

        # [barch?, out_features]
        out = self.pool(out).squeeze()

        out = out.argmax(0)
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
