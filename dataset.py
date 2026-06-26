import torch
import pandas as pd
from pathlib import Path
import torchaudio
from utils import mel_spectrogram, EMO_EFATURES
from typing import Tuple, List

# TODO: tran dataset and valid dataset
_THIS_DIR = Path(__file__).parent.resolve()
_DATA_DIR = _THIS_DIR / "EmoSpeech-0020"
EmoDB = csv = pd.read_csv("./EmoSpeech-0020/0020.txt", sep="\t", header=None)

_natural = "Neutral"
_surprise = "Surprise"
_angry = "Angry"
_happy = "Happy"
_sad = "Sad"


def max_2_div(num: int) -> int:
    out = 1
    while num % 2 == 0:
        out *= 2
        num //= 2
    return out


class EmoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        n_fft: int = 1024,
        n_mels: int = 80,
        sample_rate: int = 16000,
        hop_length: int = 256,
        win_length: int = 1024,
        f_min: float = 0.0,
        f_max: int = 8000,
    ):
        self.emodb = data
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.mel_div = max_2_div(n_mels)

    @property
    def min_div(self) -> int:
        return self.mel_div

    @property
    def mels_count(self) -> int:
        return self.n_mels

    @property
    def emo_features(self) -> int:
        return EMO_EFATURES

    def __len__(self) -> int:
        return self.emodb.shape[0]

    def get_mel(self, file: str | Path) -> torch.Tensor:
        audio, sr = torchaudio.load(file)
        assert sr == self.sample_rate
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
        )
        return mel

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        wav_file: str = self.emodb[0][index]
        emo: str = self.emodb[2][index]
        emo_file = _DATA_DIR / emo / f"{wav_file}.wav"
        mel_data = self.get_mel(emo_file)
        emo_data: torch.Tensor
        match emo:
            case "Surprise":
                emo_data = torch.tensor([0, 1, 0, 0, 0])
            case "Angry":
                emo_data = torch.tensor([0, 0, 1, 0, 0])
            case "Happy":
                emo_data = torch.tensor([0, 0, 0, 1, 0])
            case "Sad":
                emo_data = torch.tensor([0, 0, 0, 0, 1])
            case _:
                emo_data = torch.tensor([1, 0, 0, 0, 0])
        return emo_data, mel_data


class EmoBatchCollate(object):
    def __init__(self, min_div: int, emo_features: int, n_mels: int):
        self.min_div = min_div
        self.emo_features = emo_features
        self.n_mels = n_mels
        super().__init__()

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_len = len(batch)
        mel_max_len = max(data[1].shape[-1] for data in batch)

        for _ in range(self.min_div):
            if mel_max_len % self.min_div == 0:
                break
            mel_max_len += 1
        assert mel_max_len % self.min_div == 0
        emo_data = torch.zeros((batch_len, self.emo_features), dtype=torch.long)
        mel_data = torch.zeros(
            (batch_len, self.n_mels, mel_max_len), dtype=torch.float32
        )

        for i, item in enumerate(batch):
            emo, mel = item
            emo_data[i] = emo
            mel_data[i, :, : mel.shape[-1]] = mel
        return emo_data, mel_data


if __name__ == "__main__":
    dataset = EmoDataset(EmoDB, n_fft=1024, n_mels=80)
    from torch.utils.data import DataLoader

    batch_collate = EmoBatchCollate(
        dataset.min_div, dataset.emo_features, dataset.mels_count
    )
    loader = DataLoader(
        dataset=dataset, shuffle=True, batch_size=10, collate_fn=batch_collate
    )

    count = 0

    for emo_data, mel_data in loader:
        print(emo_data)
        print(mel_data.shape)
        print(emo_data.argmax(dim=1))
        count += 1
        if count >= 3:
            break
