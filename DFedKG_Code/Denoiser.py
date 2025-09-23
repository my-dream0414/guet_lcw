# -*- codeing = utf-8 -*-
# @Time : 2025/9/18 11:19
# @Author : Luo_CW
# @File : Denoiser.py
# @Software : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========== Diffusion Schedule ==========
def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """
    线性beta调度（前向扩散的噪声强度）
    """
    return torch.linspace(beta_start, beta_end, T)

# ========== Conditional Denoiser ==========
class ConditionalDenoiser(nn.Module):
    """
    文章里的 Conditional Denoiser:
    输入: (Xt, Xh, Xr, timestep)
    输出: 预测噪声 ε
    """
    def __init__(self, embed_dim, hidden_dim=1600, num_layers=2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition_mlp = nn.Linear(2 * embed_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(PointConvBlock(hidden_dim))

        self.out = nn.Linear(hidden_dim, embed_dim)

    def forward(self, Xt, Xh, Xr, t):
        # Xt: [B, D], Xh, Xr: [B, D], t: [B]
        cond = torch.cat([Xh, Xr], dim=-1)  # 拼接 head+relation
        cond = self.condition_mlp(cond)     # [B, hidden_dim]

        t_embed = self.time_mlp(t.unsqueeze(-1).float())  # [B, hidden_dim]

        h = cond + t_embed + Xt  # 初始输入

        for layer in self.layers:
            h = layer(h)

        return self.out(h)


class PointConvBlock(nn.Module):
    """
    简化版 PointConv + 残差结构
    （论文里提到用 PointConv 提取局部结构特征）
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        h = self.norm(h + residual)  # 残差连接
        return F.relu(h)