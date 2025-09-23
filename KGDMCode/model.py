# ---------------------
# Model components
# ---------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    def __init__(self, T: int, dim: int, out_dim: int):
        super().__init__()
        self.emb = nn.Embedding(T, dim)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, t: torch.Tensor):
        # t: (B,) 值在 [0, T-1] 范围内
        return self.proj(self.emb(t))

class ScoringModule(nn.Module):
    def __init__(self, mode: str = 'add'):
        super().__init__()
        assert mode in ['add', 'hadamard']
        self.mode = mode

    def forward(self, Xh, Xr):
        if self.mode == 'add':
            return Xh + Xr
        else:
            return Xh * Xr  # Hadamard 积

class CEDenoiserBlock(nn.Module):
    """一个简单的 MLP 块，具有 LN-pre、残差和残差前的可学习尺度 alpha（Peebles & Xie 2022）。"""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.alpha = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Pre-LN
        y = self.ln1(x)
        y = F.gelu(self.fc1(y))
        y = self.fc2(y)
        # 残差前缩放
        y = y * self.alpha
        return x + y

class CEDenoiser(nn.Module):
    def __init__(self, vec_dim: int, hidden: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([CEDenoiserBlock(vec_dim, hidden) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(vec_dim)
        self.out = nn.Linear(vec_dim, vec_dim)  # predict epsilon

    def forward(self, x_t, t_emb, cond):
        # 通过小型适配器连接并投影到 vec_dim 以提高稳定性
        # 输入的形状均为 (B,D)。我们将这些输入连接起来，然后使用线性回归得到 D。
        z = torch.cat([x_t, t_emb, cond], dim=-1)
        if not hasattr(self, 'adapter'):
            self.adapter = nn.Linear(z.size(-1), x_t.size(-1)).to(z.device)
        h = self.adapter(z)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        eps = self.out(h)
        return eps