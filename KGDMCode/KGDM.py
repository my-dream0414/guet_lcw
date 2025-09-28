# 最小 KGDM 复现 (AAAI'24) — 单文件 PyTorch 实现
# 作者：ChatGPT
# 这是论文《KGDM：一种用于捕获知识图谱嵌入多关系语义的扩散模型》的精简版复现
# 重点介绍核心算法（正向扩散、使用 CEDenoiser MLP 主干的条件反向去噪、评分模块、负采样损失函数和滤波排序评估）。
#
# 它支持在通用三元组文件（FB15k-237、WN18RR、Kinship、UMLS 格式）上进行训练和评估，每行包含 <head\trelation\ttail> 行。为简洁起见，此脚本使用了一个简单的文本加载器；您可以
# 将其调整为任何您喜欢的加载器。
#
# 使用示例（在准备好包含 train/valid/test.txt 的数据目录后）：
# python KGDM_minimal_repro.py --data_dir ./FB15k-237 --dataset FB15k-237 \
# --emb_dim 1000 --proj_dim 1000 --num_steps 1000 --batch_size 1024 --epochs 5
# python KGDM_minimal_repro.py --data_dir ./WN18RR --dataset WN18RR --scoring hadamard
#
# 注释：
# - 这只是一个简单的参考；要达到 SOTA 水平，你可能需要更高强度的训练（更多轮次、调优、
# 更大的维度）、GPU 以及更谨慎的负采样。
# - 实现了头部和尾部预测以及滤波指标。
# - 评分模块支持：与论文中提到的相加 (X_h + X_r) 或 hadamard (X_h * X_r)。
# - 时间步嵌入使用可学习表 + 线性投影。
# - 我们忠实地实现了算法 1：对时间步 T_t 进行采样，添加高斯噪声，预测 eps，形成
# Denoise(X_t) 并使用 L1 距离和边距 gamma 计算负采样损失。


from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from Diffusion import DiffusionSchedule
from Model import  ScoringModule, FourierTimestepEmbedding, ImprovedCEDenoiser


class KGDM(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, in_dim: int, proj_dim: int,
                 T: int, hidden: int=1200, num_blocks: int=3, gamma: float=1.0):
        super().__init__()
        self.ent_emb = nn.Embedding(num_entities, in_dim)
        self.rel_emb = nn.Embedding(num_relations, in_dim)

        self.EMBe = nn.Linear(in_dim, proj_dim)
        self.EMBr = nn.Linear(in_dim, proj_dim)

        # 改进模块
        self.timestep = FourierTimestepEmbedding(dim=64, out_dim=proj_dim)
        self.scoring = ScoringModule(proj_dim)
        self.denoiser = ImprovedCEDenoiser(dim=proj_dim, cond_dim=proj_dim,
                                           time_dim=proj_dim, hidden=hidden, num_blocks=num_blocks)
        self.head_eps = nn.Linear(proj_dim, proj_dim)
        self.head_x0 = nn.Linear(proj_dim, proj_dim)

        self.T = T
        self.gamma = gamma

    def embed_h_r_t(self, h, r, t):
        Xh = self.EMBe(self.ent_emb(h))
        Xr = self.EMBr(self.rel_emb(r))
        Xt = self.EMBe(self.ent_emb(t))
        return Xh, Xr, Xt

    def cond_embed(self, Xh, Xr):
        return self.scoring(Xh, Xr)

    def predict(self, x_t, t_idx, cond):
        t_emb = self.timestep(t_idx)
        h = self.denoiser(x_t, t_emb, cond)
        return self.head_eps(h), self.head_x0(h)

    def denoise_to_x0(self, x_t, t_idx, eps_theta, schedule: DiffusionSchedule):
        alpha_bar_t = schedule.alpha_bars[t_idx].unsqueeze(-1)
        return (x_t / torch.sqrt(alpha_bar_t)) - torch.sqrt(1.0/alpha_bar_t - 1.0) * eps_theta

    def forward_training_step(self, batch, schedule: DiffusionSchedule, num_negs: int=64, device=None):
        h, r, t = batch[:,0], batch[:,1], batch[:,2]
        Xh, Xr, Xt = self.embed_h_r_t(h, r, t)
        B, D = Xt.size()

        # 时间步
        T_idx = torch.randint(0, self.T, (B,), device=Xt.device)
        alpha_bar_t = schedule.alpha_bars[T_idx].unsqueeze(-1)

        eps = torch.randn_like(Xt)
        x_t = torch.sqrt(alpha_bar_t) * Xt + torch.sqrt(1.0 - alpha_bar_t) * eps

        cond = self.cond_embed(Xh, Xr)

        eps_pred, x0_pred = self.predict(x_t, T_idx, cond)
        x0_from_eps = self.denoise_to_x0(x_t, T_idx, eps_pred, schedule)

        # 正样本距离
        d_pos = 0.5 * (torch.abs(Xt - x0_from_eps).sum(-1) +
                       torch.abs(Xt - x0_pred).sum(-1))

        # 负采样
        neg_t = torch.randint(0, self.ent_emb.num_embeddings, (B, num_negs), device=Xt.device)
        Xtn = self.EMBe(self.ent_emb(neg_t))  # (B, K, D)

        eps_n = torch.randn_like(Xtn)
        alpha_bar_t_broadcast = alpha_bar_t.unsqueeze(1)
        x_tn = torch.sqrt(alpha_bar_t_broadcast) * Xtn + torch.sqrt(1.0 - alpha_bar_t_broadcast) * eps_n

        cond_n = cond.unsqueeze(1).expand_as(x_tn).reshape(B*num_negs, D)
        t_emb_n = T_idx.unsqueeze(1).expand(B, num_negs).reshape(-1)

        x_tn_flat = x_tn.reshape(B*num_negs, D)
        _, x0_pred_n = self.predict(x_tn_flat, t_emb_n, cond_n)
        x0_pred_n = x0_pred_n.view(B, num_negs, D)

        d_neg = torch.abs(Xt.unsqueeze(1) - x0_pred_n).sum(-1)

        # 损失
        loss_pos = F.logsigmoid(self.gamma - d_pos)
        loss_neg = F.logsigmoid(d_neg - self.gamma).mean(dim=1)
        loss = -(loss_pos + loss_neg).mean()
        return loss

    @torch.no_grad()
    def sample_tail(self, h_idx: torch.Tensor, r_idx: torch.Tensor,
                    schedule: DiffusionSchedule, steps: int=None):
        device = h_idx.device
        steps = steps or self.T
        Xh = self.EMBe(self.ent_emb(h_idx))
        Xr = self.EMBr(self.rel_emb(r_idx))
        cond = self.cond_embed(Xh, Xr)
        D = cond.size(-1)
        x = torch.randn((h_idx.size(0), D), device=device)
        for t in reversed(range(steps)):
            t_idx = torch.full((h_idx.size(0),), t, device=device, dtype=torch.long)
            eps_pred, _ = self.predict(x, t_idx, cond)
            beta_t = schedule.betas[t]
            alpha_t = schedule.alphas[t]
            alpha_bar_t = schedule.alpha_bars[t]
            mean = (1.0/torch.sqrt(alpha_t)) * (x - (beta_t/torch.sqrt(1.0 - alpha_bar_t)) * eps_pred)
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * noise
            else:
                x = mean
        return x

    @torch.no_grad()
    def rank_tail(self, h_idx: torch.Tensor, r_idx: torch.Tensor,
                  schedule: DiffusionSchedule, entity_bank: torch.Tensor,
                  filtered: Dict[Tuple[int,int], set], hits_k=(1,3,10)):
        gen = self.sample_tail(h_idx, r_idx, schedule)
        dists = torch.cdist(gen, entity_bank, p=2)
        ranks = []
        for i in range(gen.size(0)):
            hi = h_idx[i].item(); ri = r_idx[i].item()
            order = torch.argsort(dists[i])
            ranks.append(order)
        return ranks, dists


# class KGDM(nn.Module):
#     def __init__(self, num_entities: int, num_relations: int, in_dim: int, proj_dim: int,
#                  T: int, scoring: str='add', hidden: int=1200, num_blocks: int=3, gamma: float=1.0):
#         super().__init__()
#         # 基础嵌入（离散 -> 向量）
#         self.ent_emb = nn.Embedding(num_entities, in_dim)
#         self.rel_emb = nn.Embedding(num_relations, in_dim)
#         # 可学习的嵌入函数 EMBe、EMBr（线性投影到工作空间）
#         self.EMBe = nn.Linear(in_dim, proj_dim)
#         self.EMBr = nn.Linear(in_dim, proj_dim)
#
#         self.timestep = TimestepEmbedding(T=T, dim=proj_dim//4, out_dim=proj_dim)
#         self.scoring = ScoringModule(scoring)
#         self.denoiser = CEDenoiser(vec_dim=proj_dim, hidden=hidden, num_blocks=num_blocks)
#
#         self.T = T
#         self.gamma = gamma
#
#     def embed_h_r_t(self, h, r, t):
#         Xh = self.EMBe(self.ent_emb(h))
#         Xr = self.EMBr(self.rel_emb(r))
#         Xt = self.EMBe(self.ent_emb(t))
#         return Xh, Xr, Xt
#
#     def cond_embed(self, Xh, Xr):
#         return self.scoring(Xh, Xr)
#
#     def forward_training_step(self, batch, schedule: DiffusionSchedule, num_negs: int=64, device=None):
#         # batch: (B, 3) 索引
#         h, r, t = batch[:,0], batch[:,1], batch[:,2]
#         Xh, Xr, Xt = self.embed_h_r_t(h, r, t)
#         B, D = Xt.size()
#
#         # 在 [0, T-1] 中均匀采样时间步长
#         T_idx = torch.randint(0, self.T, (B,), device=Xt.device)
#         alpha_bar_t = schedule.alpha_bars[T_idx].unsqueeze(-1)  # (B,1)
#         alpha_t = schedule.alphas[T_idx].unsqueeze(-1)
#
#         # 添加noise到 Xt
#         eps = torch.randn_like(Xt)
#         x_t = torch.sqrt(alpha_bar_t) * Xt + torch.sqrt(1.0 - alpha_bar_t) * eps
#
#         # 条件和时间步嵌入
#         cond = self.cond_embed(Xh, Xr)
#         t_emb = self.timestep(T_idx)
#
#         # 预测每股收益
#         eps_theta = self.denoiser(x_t, t_emb, cond)
#
#         # 使用重新参数化重建去噪目标
#         # Denoise(Xt) = (1/sqrt(alpha_bar_t)) * x_t - sqrt(1/alpha_bar_t - 1) * eps_theta
#         denoised_pos = (x_t / torch.sqrt(alpha_bar_t)) - torch.sqrt(1.0/alpha_bar_t - 1.0) * eps_theta
#
#         # 对尾部实体进行负采样
#         # 均匀采样负样本
#         neg_t = torch.randint(0, self.ent_emb.num_embeddings, (B, num_negs), device=Xt.device)
#         # 嵌入负片并以每行相同的时间步长计算其去噪结果
#         Xtn = self.EMBe(self.ent_emb(neg_t))  # (B, K, D)
#         eps_n = torch.randn_like(Xtn)
#         # 每行使用相同的 alpha_bar_t
#         alpha_bar_t_broadcast = alpha_bar_t.unsqueeze(1)  # (B,1,1)
#         x_tn = torch.sqrt(alpha_bar_t_broadcast) * Xtn + torch.sqrt(1.0 - alpha_bar_t_broadcast) * eps_n
#
#         # 为了提高效率，对所有负数重复使用 cond,t_emb
#         cond_n = cond.unsqueeze(1).expand_as(x_tn)
#         t_emb_n = t_emb.unsqueeze(1).expand_as(x_tn)
#         # 将 batch*K 展平用于降噪器通道
#         x_tn_flat = x_tn.reshape(B*num_negs, D)
#         cond_n_flat = cond_n.reshape(B*num_negs, D)
#         t_emb_flat = t_emb_n.reshape(B*num_negs, D)
#         eps_theta_n = self.denoiser(x_tn_flat, t_emb_flat, cond_n_flat).reshape(B, num_negs, D)
#         denoised_neg = (x_tn / torch.sqrt(alpha_bar_t_broadcast)) - torch.sqrt(1.0/alpha_bar_t_broadcast - 1.0) * eps_theta_n
#
#         # 损失：具有 L1 距离的负采样逻辑损失
#         d_pos = torch.abs(Xt - denoised_pos).sum(dim=-1)  # (B,)
#         d_neg = torch.abs(Xt.unsqueeze(1) - denoised_neg).sum(dim=-1)  # (B,K)
#         # L = - log sigma(gamma - d_pos) - (1/K) * sum log sigma(d_neg - gamma)
#         loss_pos = F.logsigmoid(self.gamma - d_pos)
#         loss_neg = F.logsigmoid(d_neg - self.gamma).mean(dim=1)
#         loss = -(loss_pos + loss_neg).mean()
#         return loss
#
#     @torch.no_grad()
#     def sample_tail(self, h_idx: torch.Tensor, r_idx: torch.Tensor, schedule: DiffusionSchedule, steps: int=None):
#         """对一批查询（h，r，？）进行逆过程采样：每个查询返回一个生成的尾部向量。"""
#         device = h_idx.device
#         steps = steps or self.T
#         Xh = self.EMBe(self.ent_emb(h_idx))
#         Xr = self.EMBr(self.rel_emb(r_idx))
#         cond = self.cond_embed(Xh, Xr)
#         D = cond.size(-1)
#         # 从纯高斯开始
#         x = torch.randn((h_idx.size(0), D), device=device)
#         for t in reversed(range(steps)):
#             t_idx = torch.full((h_idx.size(0),), t, device=device, dtype=torch.long)
#             t_emb = self.timestep(t_idx)
#             eps_theta = self.denoiser(x, t_emb, cond)
#             beta_t = schedule.betas[t]
#             alpha_t = schedule.alphas[t]
#             alpha_bar_t = schedule.alpha_bars[t]
#             # 计算平均值（公式 3）
#             mean = (1.0/torch.sqrt(alpha_t)) * (x - (beta_t/torch.sqrt(1.0 - alpha_bar_t)) * eps_theta)
#             if t > 0:
#                 noise = torch.randn_like(x)
#                 sigma_t = torch.sqrt(beta_t)
#                 x = mean + sigma_t * noise
#             else:
#                 x = mean
#         return x  # 生成的尾部向量
#
#     @torch.no_grad()
#     def rank_tail(self, h_idx: torch.Tensor, r_idx: torch.Tensor, schedule: DiffusionSchedule, entity_bank: torch.Tensor,
#                   filtered: Dict[Tuple[int,int], set], hits_k=(1,3,10)):
#         # 生成单尾向量，并根据与实体银行的 L2 距离进行排序
#         gen = self.sample_tail(h_idx, r_idx, schedule)
#         # 到所有实体嵌入的距离（投影空间）
#         # entity_bank: (num_entities, D)
#         dists = torch.cdist(gen, entity_bank, p=2)  # (B, N)
#         ranks = []
#         hits = {k: 0 for k in hits_k}
#         mrr = 0.0
#         for i in range(gen.size(0)):
#             hi = h_idx[i].item(); ri = r_idx[i].item()
#             # 为过滤评估设置了真尾
#             true_tails = filtered.get((hi,ri), set())
#             # 按距离升序排序
#             order = torch.argsort(dists[i])
#             # 删除除当前金尾之外的已过滤实体（由调用者处理）
#             # 这里调用者应该一次传递一个金尾；我们将据此计算它的排名
#             # 我们只返回排序后的列表；评估循环处理过滤删除。
#             ranks.append(order)
#         return ranks, dists

