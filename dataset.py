import torch
import pandas as pd
from pathlib import Path
import torchaudio
from utils import mel_spectrogram
from typing import Tuple

_THIS_DIR = Path(__file__).parent.resolve()
_DATA_DIR = _THIS_DIR / "EmoSpeech-0020"
_db = csv = pd.read_csv("./EmoSpeech-0020/0020.txt", sep="\t", header=None)

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
        sample_rate: int = 22050,
        hop_length: int = 256,
        win_length: int = 1024,
        f_min: float = 0.0,
        f_max: int = 8000,
    ):
        assert max_2_div(n_fft) >= max_2_div(n_mels)
        self.emodb = data
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max

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


if __name__ == "__main__":
    dataset = EmoDataset(_db, n_fft=1024, n_mels=80)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset=dataset, shuffle=True, batch_size=10)

    # TODO: add a arrangement
    for emo_data, mel_data in loader:
        print(emo_data)
        break
