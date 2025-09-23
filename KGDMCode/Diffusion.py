# ---------------------
# Diffusion schedule helpers
# ---------------------
import torch
from dataclasses import dataclass

def make_beta_schedule(T: int, beta_start: float=1e-4, beta_end: float=2e-2):
    return torch.linspace(beta_start, beta_end, T)

@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    @classmethod
    def create(cls, T: int, device: torch.device):
        betas = make_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return cls(betas, alphas, alpha_bars)