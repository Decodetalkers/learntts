from typing import Tuple
import torch
import torch.nn as nn


def get_noise(
    t: torch.Tensor, beta_init: float, beta_term: float, cumulative: bool = False
) -> torch.Tensor:
    if cumulative:
        noise = beta_init * t + 0.5 * (beta_term - beta_init) * (t**2)
    else:
        noise = beta_init + (beta_term - beta_init) * t
    return noise


class Diffusion(nn.Module):
    def __init__(self, beta_min: float = 0.05, beta_max: int | float = 20, cumulative: bool = True):
        super().__init__()
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
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.forward(x0, t), t
