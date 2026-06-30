import torch
from typing import Optional

EMO_FEATURES = 5


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
    from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

    spec, _ = mel_spectogram(
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
