import torch
import torch.nn as nn
from transformers import HubertModel
from typing import Optional

EMO_FEATURES = 5

# frames 768
HUBEDRT_FEATURES = 768

class HubertEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()

        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")

        if freeze:
            self.hubert.requires_grad_(False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, samples]

        output = self.hubert(input_values=waveform)

        # [B, frames, 768]
        features = output.last_hidden_state

        # [channel, frames, 768]
        return features

    @torch.no_grad()
    def predict(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.forward(waveform)

_ENCODER = HubertEncoder()

def hubert_features(y: torch.Tensor) -> torch.Tensor:
    return _ENCODER.predict(y).squeeze()

def yin(
    y: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    win_length: int,
    f_min: float,
    f_max: Optional[float],
) -> torch.Tensor:
    f_max = f_max if f_max is not None else float(sample_rate // 2)
    import librosa

    yin = librosa.yin(
        y=y.numpy(),
        fmin=440,
        fmax=f_max,
        sr=sample_rate,
        hop_length=hop_length,
        frame_length=win_length,
    )
    return torch.from_numpy(yin).squeeze()


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    hop_length: int,
    win_length: int,
    f_min: float,
    f_max: Optional[float],
) -> torch.Tensor:
    from speechbrain.lobes.models.FastSpeech2 import mel_spectrogram

    spec, _ = mel_spectrogram(
        audio=y.squeeze(),
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        f_min=f_min,
        f_max=f_max,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True,
    )
    return spec


def fix_len_compatibility(length: int, num_downsamplings_in_unet: int = 4) -> int:
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1
