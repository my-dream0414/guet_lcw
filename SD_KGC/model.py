# model.py
# SD-KGC 模型：HighOrderEncoder + Relation-guided aggregator + Residual MLP denoiser diffusion
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_NEIGH = 50  # 若实体邻居很多，训练中可采样

class RelationAwareAggregator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_e = nn.Linear(in_dim, out_dim, bias=False)
        self.W_r = nn.Linear(in_dim, out_dim, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.att = nn.Linear(out_dim*2, 1, bias=False)

    def forward(self, node_emb, neigh_embs, neigh_rels, mask=None):
        # node_emb: (B,d)
        # neigh_embs: (B,K,d), neigh_rels: (B,K,d)
        h = self.W_e(node_emb).unsqueeze(1)               # (B,1,out)
        neigh_h = self.W_e(neigh_embs)                    # (B,K,out)
        rel_h = self.W_r(neigh_rels)                      # (B,K,out)
        nb_repr = neigh_h + rel_h                         # (B,K,out)
        score_input = torch.cat([h.expand_as(nb_repr), nb_repr], dim=-1)  # (B,K,2*out)
        scores = self.att(self.leaky(score_input)).squeeze(-1)            # (B,K)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        alpha = torch.softmax(scores, dim=-1)
        out = torch.sum(alpha.unsqueeze(-1) * nb_repr, dim=1)             # (B,out)
        return out, alpha

class HighOrderEncoder(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim, max_neigh=MAX_NEIGH):
        super().__init__()
        self.ent_emb = nn.Embedding(num_entities, emb_dim)
        self.rel_emb = nn.Embedding(num_relations, emb_dim)
        self.agg1 = RelationAwareAggregator(emb_dim, emb_dim)
        self.agg2 = RelationAwareAggregator(emb_dim, emb_dim)
        self.res_lambda = 0.2
        self.max_neigh = max_neigh

        # init
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, nodes, neighbor_sampler):
        """
        nodes: (B,) long tensor of entity ids
        neighbor_sampler: callable that given list of node ids returns
                          neigh_ids (B,K) and neigh_rels (B,K) and mask (B,K)
                          This keeps IO flexible (precomputed or on-the-fly).
        """
        device = nodes.device
        h0 = self.ent_emb(nodes)  # (B,d)
        neigh_ids, neigh_rels, mask = neighbor_sampler(nodes.tolist(), self.max_neigh, device)
        # neighbor embeddings
        neigh_embs = self.ent_emb(neigh_ids)    # (B,K,d)
        neigh_rel_embs = self.rel_emb(neigh_rels)
        out1, _ = self.agg1(h0, neigh_embs, neigh_rel_embs, mask=mask)
        h1 = (1 - self.res_lambda) * h0 + self.res_lambda * out1
        out2, _ = self.agg2(h1, neigh_embs, neigh_rel_embs, mask=mask)
        h2 = (1 - self.res_lambda) * h1 + self.res_lambda * out2
        return h2

class ResidualMLPDenoiser(nn.Module):
    """
    残差 MLP 去噪器：输入 (h_t, cond=head+rel, t_scalar) -> 预测 noise
    """
    def __init__(self, emb_dim, hidden_dim=512, num_layers=4, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = emb_dim + emb_dim*2 + 1  # ht + head + rel + t_scalar (if cond is head+rel)
        # We'll pass cond as head+rel concatenated, so cond_dim = 2*emb_dim
        # but to keep API flexible we accept ht + cond + t_feat concatenation at call site.
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i==0 else hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, emb_dim)
        # small residual scaling
        self.res_scale = 0.1
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, ht, cond, t_scalar):
        """
        ht: (B,emb_dim)
        cond: (B, 2*emb_dim)  => head_emb || rel_emb
        t_scalar: float or (B,1) tensor normalized (0..1)
        """
        if isinstance(t_scalar, float) or isinstance(t_scalar, int):
            t_feat = torch.full((ht.size(0), 1), float(t_scalar), device=ht.device, dtype=ht.dtype)
        else:
            t_feat = t_scalar
        x = torch.cat([ht, cond, t_feat], dim=-1)
        h = self.mlp(x)
        out = self.out(h)
        return out + ht * self.res_scale

class SimpleDiffusion(nn.Module):
    def __init__(self, emb_dim, T=200, hidden_dim=512, num_layers=4, dropout=0.1):
        super().__init__()
        self.T = T
        self.emb_dim = emb_dim
        self.denoiser = ResidualMLPDenoiser(emb_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def q_sample(self, h0, t, noise=None):
        # t can be scalar int or tensor of ints (per-sample)
        if noise is None:
            noise = torch.randn_like(h0)
        a_prod = self.alphas_cumprod[t].view(-1, 1) if isinstance(t, torch.Tensor) else self.alphas_cumprod[t]
        sqrt_a = torch.sqrt(a_prod).to(h0.device)
        sqrt_om = torch.sqrt(1.0 - a_prod).to(h0.device)
        return sqrt_a * h0 + sqrt_om * noise, noise

    def predict_noise(self, ht, cond, t_scalar):
        return self.denoiser(ht, cond, t_scalar)

    def forward_loss(self, h0, cond):
        # sample t uniformly per batch
        B = h0.size(0)
        t = torch.randint(low=0, high=self.T, size=(B,), device=h0.device)
        ht, noise = self.q_sample(h0, t)
        # normalize t to [0,1]
        t_norm = t.float() / float(self.T)
        pred = self.predict_noise(ht, cond, t_norm.unsqueeze(-1))
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def generate(self, cond, device):
        # cond: (B, 2*emb_dim)
        B = cond.size(0)
        ht = torch.randn(B, self.emb_dim, device=device)
        for step in reversed(range(self.T)):
            t_norm = float(step) / float(self.T)
            pred = self.predict_noise(ht, cond, t_norm)
            alpha_t = self.alphas_cumprod[step].to(device)
            if step > 0:
                z = torch.randn_like(ht)
            else:
                z = torch.zeros_like(ht)
            coef1 = 1.0 / math.sqrt(alpha_t.item())
            coef2 = (1 - alpha_t.item()) / math.sqrt(1 - alpha_t.item())
            ht = coef1 * (ht - coef2 * pred) + math.sqrt(self.betas[step].item()) * z
        return ht

class SDKGC(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim=256, diffusion_steps=200, neighbor_sampler=None):
        super().__init__()
        self.encoder = HighOrderEncoder(num_entities, num_relations, emb_dim)
        self.rel_emb = self.encoder.rel_emb
        self.ent_emb = self.encoder.ent_emb
        self.diffusion = SimpleDiffusion(emb_dim, T=diffusion_steps)
        self.neighbor_sampler = neighbor_sampler  # callable for neighbor lookup

    def encode_head(self, h_ids):
        # h_ids: tensor (B,)
        return self.encoder(h_ids, self.neighbor_sampler)

    def forward_score_all(self, h_ids, r_ids):
        """
        给定 (h,r) batch，生成 tail embedding并与所有实体求相似度得分。
        返回 (B, num_entities) logits (未归一化)
        """
        device = h_ids.device
        h_vec = self.encode_head(h_ids)   # (B,d)
        r_vec = self.rel_emb(r_ids)       # (B,d)
        cond = torch.cat([h_vec, r_vec], dim=-1)  # (B, 2d)
        # generate tail embedding via diffusion
        gen_t = self.diffusion.generate(cond, device=device)  # (B,d)
        scores = torch.matmul(gen_t, self.ent_emb.weight.t())  # (B, E)
        return scores

    def score_positive(self, h_ids, r_ids, t_ids):
        scores = self.forward_score_all(h_ids, r_ids)
        # select the true tail logits
        inds = t_ids.view(-1,1)
        return scores.gather(1, inds).squeeze(1)
