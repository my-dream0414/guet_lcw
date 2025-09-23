# -*- codeing = utf-8 -*-
# @Time : 2025/9/18 11:19
# @Author : Luo_CW
# @File : DiffusionModel.py
# @Software : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from DFedKG_Code.Denoiser import get_beta_schedule, ConditionalDenoiser

# ========== Diffusion Model ==========
class DiffusionModel(nn.Module):
    def __init__(self, embed_dim, T=1000):
        super().__init__()
        self.T = T
        self.betas = get_beta_schedule(T)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)  # 累积乘积

        self.denoiser = ConditionalDenoiser(embed_dim)

    def q_sample(self, x0, t, noise=None):
        """
        前向扩散 q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_hat = self.alpha_hat[t].sqrt().unsqueeze(-1)
        sqrt_one_minus = (1 - self.alpha_hat[t]).sqrt().unsqueeze(-1)

        return sqrt_alpha_hat * x0 + sqrt_one_minus * noise

    def p_sample(self, Xt, Xh, Xr, t):
        """
        逆向去噪一步 p(x_{t-1}|x_t)
        """
        eps_theta = self.denoiser(Xt, Xh, Xr, t)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_hat_t = self.alpha_hat[t]

        mean = (1 / math.sqrt(alpha_t)) * (
            Xt - (beta_t / (1 - alpha_hat_t).sqrt()) * eps_theta
        )
        if t > 0:
            noise = torch.randn_like(Xt)
        else:
            noise = torch.zeros_like(Xt)
        return mean + math.sqrt(beta_t) * noise

    def forward_loss(self, x0, Xh, Xr):
        """
        训练损失: L1 或 MSE，学习预测噪声
        """
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        eps_pred = self.denoiser(xt, Xh, Xr, t)
        return F.mse_loss(eps_pred, noise)

    def sample(self, Xh, Xr):
        """
        从噪声开始逐步生成 tail embedding
        """
        B, D = Xh.size(0), Xh.size(1)
        xt = torch.randn(B, D, device=Xh.device)

        for t in reversed(range(self.T)):
            xt = self.p_sample(xt, Xh, Xr, t)
        return xt  # 作为生成的 tail embedding