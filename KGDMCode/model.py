# ---------------------
# Model components
# ---------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# class TimestepEmbedding(nn.Module):
#     def __init__(self, T: int, dim: int, out_dim: int):
#         super().__init__()
#         self.emb = nn.Embedding(T, dim)
#         self.proj = nn.Linear(dim, out_dim)
#
#     def forward(self, t: torch.Tensor):
#         # t: (B,) 值在 [0, T-1] 范围内
#         # t: (B,) 值在 [0, T-1] 范围内
#         return self.proj(self.emb(t))
#
# class ScoringModule(nn.Module):
#     def __init__(self, mode: str = 'add'):
#         super().__init__()
#         assert mode in ['add', 'hadamard']
#         self.mode = mode
#
#     def forward(self, Xh, Xr):
#         if self.mode == 'add':
#             return Xh + Xr
#         else:
#             return Xh * Xr  # Hadamard 积
#
# class ScoringModule(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.gate = nn.Linear(dim * 2, 1)
#
#     def forward(self, h, r):
#         g = torch.sigmoid(self.gate(torch.cat([h, r], dim=-1)))
#         return g * (h + r) + (1 - g) * (h * r)
#
# class CEDenoiserBlock(nn.Module):
#     """一个简单的 MLP 块，具有 LN-pre、残差和残差前的可学习尺度 alpha（Peebles & Xie 2022）。"""
#     def __init__(self, dim: int, hidden: int):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(dim)
#         self.fc1 = nn.Linear(dim, hidden)
#         self.fc2 = nn.Linear(hidden, dim)
#         self.alpha = nn.Parameter(torch.ones(dim))
#
#     def forward(self, x):
#         # Pre-LN
#         y = self.ln1(x)
#         y = F.gelu(self.fc1(y))
#         y = self.fc2(y)
#         # 残差前缩放
#         y = y * self.alpha
#         return x + y
#
# class CEDenoiser(nn.Module):
#     def __init__(self, vec_dim: int, hidden: int, num_blocks: int):
#         super().__init__()
#         self.blocks = nn.ModuleList([CEDenoiserBlock(vec_dim, hidden) for _ in range(num_blocks)])
#         self.ln = nn.LayerNorm(vec_dim)
#         self.out = nn.Linear(vec_dim, vec_dim)  # predict epsilon
#
#     def forward(self, x_t, t_emb, cond):
#         # 通过小型适配器连接并投影到 vec_dim 以提高稳定性
#         # 输入的形状均为 (B,D)。我们将这些输入连接起来，然后使用线性回归得到 D。
#         z = torch.cat([x_t, t_emb, cond], dim=-1)
#         if not hasattr(self, 'adapter'):
#             self.adapter = nn.Linear(z.size(-1), x_t.size(-1)).to(z.device)
#         h = self.adapter(z)
#         for blk in self.blocks:
#             h = blk(h)
#         h = self.ln(h)
#         eps = self.out(h)
#         return eps

class ScoringModule(nn.Module):
    """可学习门控融合 Add / Hadamard"""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, 1)

    def forward(self, h, r):
        g = torch.sigmoid(self.gate(torch.cat([h, r], dim=-1)))
        return g * (h + r) + (1 - g) * (h * r)


class FourierTimestepEmbedding(nn.Module):
    """Fourier 时间步嵌入"""
    def __init__(self, dim=64, max_freq=10000.0, out_dim=256):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(torch.arange(0, half, dtype=torch.float32) *
                          -(math.log(max_freq) / max(1, half - 1)))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, t: torch.Tensor):
        angles = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class FiLMBlock(nn.Module):
    """FiLM 条件调制"""
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.mlp = nn.Linear(cond_dim, feature_dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.mlp(cond).chunk(2, dim=-1)
        return gamma * x + beta


class PreNLinResBlock(nn.Module):
    """Pre-LN + FiLM 残差块"""
    def __init__(self, dim, hidden, cond_dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.alpha_proj = nn.Linear(cond_dim, dim)
        self.film = FiLMBlock(dim, cond_dim)

    def forward(self, x, cond):
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        h = self.film(h, cond)
        alpha = torch.sigmoid(self.alpha_proj(cond))
        return x + alpha * h


class ImprovedCEDenoiser(nn.Module):
    """改进版 CEDenoiser"""
    def __init__(self, dim, cond_dim, time_dim=256, hidden=1024, num_blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(dim + cond_dim + time_dim, dim)
        self.blocks = nn.ModuleList([PreNLinResBlock(dim, hidden, cond_dim) for _ in range(num_blocks)])
        self.out_ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x_t, t_emb, cond):
        h = self.input_proj(torch.cat([x_t, t_emb, cond], dim=-1))
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(self.out_ln(h))