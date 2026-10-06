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
