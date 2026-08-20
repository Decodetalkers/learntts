import params
import torch
import argparse
import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
from denoise import Denoise
from typing import Tuple


def get_noise(
    t: torch.Tensor, beta_init: float, beta_term: float, cumulative: bool = False
) -> torch.Tensor:
    if cumulative:
        noise = beta_init * t + 0.5 * (beta_term - beta_init) * (t**2)
    else:
        noise = beta_init + (beta_term - beta_init) * t
    return noise


class Diffusion:
    def __init__(
        self,
        beta_min: float = 0.05,
        beta_max: int | float = 20,
        cumulative: bool = True,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.cumulative = cumulative

    def forward(self, x0: torch.Tensor, t: torch.Tensor):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, self.cumulative)
        mean = x0 * torch.exp(-0.5 * cum_noise)
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt

    def diffuse(
        self, x0: torch.Tensor, offset: float = 1e-5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.rand(
            x0.shape[0], dtype=x0.dtype, device=x0.device, requires_grad=False
        )
        t = torch.clamp(t, offset, 0.3)
        return self.forward(x0, t), t


# Load a pretrained HIFIGAN Vocoder
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir="pretrained_models/tts-hifigan-ljspeech",
    run_opts={"device": "cuda:0"},
)


def dynamic_range_compression(
    x: torch.Tensor, C: int = 1, clip_val: float = 1e-5
) -> torch.Tensor:
    """Dynamic range compression for audio signals"""
    return torch.log(torch.clamp(x, min=clip_val) * C)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="path to a checkpoint of Denoise",
    )
    args = parser.parse_args()
    signal, rate = torchaudio.load("./EmoSpeech-0020/Angry/0020_000351.wav")
    resample = torchaudio.transforms.Resample(rate, 22050)

    signal = resample(signal)
    spectrogram, _ = mel_spectogram(
        audio=signal.squeeze(),
        sample_rate=22050,
        hop_length=256,
        win_length=None,
        n_mels=80,
        n_fft=1024,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True,
    )
    diffusion = Diffusion()

    spectrogram = spectrogram.unsqueeze(dim=0)

    spectrogram, t = diffusion.diffuse(spectrogram)

    denoise = Denoise().to(params.device)

    denoise.load_state_dict(
        torch.load(args.checkpoint, map_location=lambda loc, _: loc)
    )

    spectrogram = denoise.denoise(spectrogram.cuda())

    spectrogram = spectrogram.squeeze()
    waveforms = hifi_gan.decode_batch(spectrogram)

    # Save the reconstructed audio as a waveform
    torchaudio.save("waveform_reconstructed.wav", waveforms.squeeze(1).cpu(), 22050)
