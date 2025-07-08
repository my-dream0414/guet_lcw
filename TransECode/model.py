# -*- codeing = utf-8 -*-
# @Time : 2025/7/7 15:27
# @Author : Luo_CW
# @File : model.py
# @Software : PyCharm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class TransE(nn.Module):
    def __init__(self ,entity_count ,relation_count, device, norm=1, dim=100, margin=1.0):
        super(TransE, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.norm = norm
        self.dim = dim
        # 实体和关系嵌入
        self.entities_emb = self._init_enitity_emb()
        self.relations_emb = self._init_relation_emb()
        # 损失函数
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')


    def _init_enitity_emb(self):
        entities_emb = nn.Embedding(num_embeddings=self.entity_count + 1,
                                    embedding_dim=self.dim,
                                    padding_idx=self.entity_count)
        uniform_range = 6 / np.sqrt(self.dim)
        entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return entities_emb

    def _init_relation_emb(self):
        relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1,
                                     embedding_dim=self.dim,
                                     padding_idx=self.relation_count)
        uniform_range = 6 / np.sqrt(self.dim)
        relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        # -1 表示避免 OOV 向量为 nan
        relations_emb.weight.data[:-1, :].div_(relations_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return relations_emb

    def forward(self,
                positive_triplets: torch.Tensor,
                negative_triplets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """正负三元组的前向传播。"""
        # 实体规范化 (L2)
        with torch.no_grad():
            self.entities_emb.weight.data[:-1] = F.normalize(
                self.entities_emb.weight.data[:-1], p=2, dim=1
            )

        # 计算距离
        positive_distances = self._distance(positive_triplets)
        negative_distances = self._distance(negative_triplets)

        # 计算损失
        loss = self.loss(positive_distances, negative_distances)
        return loss, positive_distances, negative_distances

    def predict(self, triplets: torch.Tensor) -> torch.Tensor:
        """预测给定三元组的差异性得分。"""
        return self._distance(triplets)

    def loss(self, positive_distances: torch.Tensor, negative_distances: torch.Tensor) -> torch.Tensor:
        target = torch.tensor([-1], dtype=torch.int64, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def _distance(self, triplets: torch.Tensor) -> torch.Tensor:
        """计算三元组的 TransE 距离。"""
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        return (
                self.entities_emb(heads) +
                self.relations_emb(relations) -
                self.entities_emb(tails)
                .norm(p=self.norm, dim=1))
