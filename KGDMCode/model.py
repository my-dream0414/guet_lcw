# -*- codeing = utf-8 -*-
# @Time : 2025/7/21 15:32
# @Author : Luo_CW
# @File : model.py
# @Software : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

class KGDM(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim, timesteps):
        super(KGDM, self).__init__()
        self.emb_dim = emb_dim
        self.timesteps = timesteps  # T

        # 实体和关系嵌入
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)

        # 去噪网络（f_theta）：使用一个简单的 MLP
        self.denoiser = nn.Sequential(
            nn.Linear(emb_dim + 1, emb_dim),  # +1 for timestep embedding
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # 时间步嵌入（可选）
        self.timestep_emb = nn.Embedding(timesteps + 1, 1)

    def forward_diffusion(self, r, t):
        """
        正向扩散：在 r 上添加噪声生成 r_t
        """
        noise = torch.randn_like(r)
        alpha_t = 1 - t / self.timesteps  # 简化的线性噪声调度
        r_t = alpha_t * r + (1 - alpha_t) * noise
        return r_t, noise

    def denoise(self, r_t, t):
        """
        用于噪声预测的去噪网络 f_theta(r_t, t)
        """
        t_emb = self.timestep_emb(t).squeeze(-1)  # shape: (batch, 1)
        x = torch.cat([r_t, t_emb], dim=-1)
        pred_noise = self.denoiser(x)
        return pred_noise

    def score(self, h, r_t, t_emb, tail):
        """
        使用 TransE 风格的 score 函数
        f(h, r_t, t) = - || h + r_t - t ||
        """
        return -torch.norm(h + r_t - tail, p=2, dim=1)

    def forward(self, h_idx, r_idx, t_idx, time_step):
        """
        :param h_idx: head entity index (batch,)
        :param r_idx: relation index (batch,)
        :param t_idx: tail entity index (batch,)
        :param time_step: 当前扩散步骤 (batch,)
        """
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        tail = self.entity_emb(t_idx)

        r_t, noise = self.forward_diffusion(r, time_step)  # 正向扩散生成 r_t
        pred_noise = self.denoise(r_t, time_step)          # 去噪预测
        loss_denoise = F.mse_loss(pred_noise, noise)       # 去噪损失

        score_pos = self.score(h, r_t, time_step, tail)    # 正样本打分

        return loss_denoise, score_pos
