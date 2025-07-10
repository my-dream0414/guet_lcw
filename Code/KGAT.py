# -*- codeing = utf-8 -*-
# @Time : 2025/7/1 21:08
# @Author : Luo_CW
# @File : KGAT.py
# @Software : PyCharm
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import edge_softmax_fix

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):
    '''
    通用的图聚合器
    用于知识图谱中节点特征的聚合更新。
    支持三种聚合方式：
        GCN
        GraphSAGE
        Bu-Interation
    整合路DGL的图计算功能
    核心功能：
        输入：图结构+节点当前嵌入
        输出：聚合后的新节点表示（形状：[n_node,out_dim]）
        关键操作：
            通过边注意力权重（att）聚合邻居信息
            使用不同策略（如GCN、GraphSAGE）融合中心点与邻居信息
            应用激活函数和Dropout正则化
    '''

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        '''
        :param in_dim: 输入特征维度
        :param out_dim: 输出特征维度
        :param dropout: Dropout概率
        :param aggregator_type: 聚合类型（gcn/graphsage/bi-interaction）
        '''
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        # # GCN单层线性变换
        if aggregator_type == 'gcn':
            self.W = nn.Linear(self.in_dim, self.out_dim)       # W in Equation (6)
        # GraphSAGE拼接后变换
        elif aggregator_type == 'graphsage':
            self.W = nn.Linear(self.in_dim * 2, self.out_dim)   # W in Equation (7)

        elif aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in Equation (8)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in Equation (8)
        else:
            raise NotImplementedError  # 抛出异常

        self.activation = nn.LeakyReLU()

    def forward(self, mode, g, entity_embed):
        '''
        # 模式选择：训练用dgl原生sum（快速），预测用自定义sum（确定性强）
        :param mode:
        :param g:
        :param entity_embed:
        :return:
        '''

        # 准备数据
        g = g.local_var()
        g.ndata['node'] = entity_embed

        # Equation (3) & (10)
        # DGL: dgl-cu10.1(0.5.3)
        # 使用 `dgl.function.sum` 时会得到不同的结果，随机性源于 `atomicAdd`
        # 训练模型时使用 `dgl.function.sum` 来加速
        # 使用自定义函数来确保预测时的确定性行为
        if mode == 'predict':
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), lambda nodes: {'N_h': torch.sum(nodes.mailbox['side'], 1)})
        else:
            # side = 节点编码向量 @ 知识意识注意， N_h = sum(side)
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), dgl.function.sum('side', 'N_h'))

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            out = self.activation(self.W(g.ndata['node'] + g.ndata['N_h']))                         # (n_users + n_entities, out_dim)

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            out = self.activation(self.W(torch.cat([g.ndata['node'], g.ndata['N_h']], dim=1)))      # (n_users + n_entities, out_dim)

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            out1 = self.activation(self.W1(g.ndata['node'] + g.ndata['N_h']))                       # (n_users + n_entities, out_dim)
            out2 = self.activation(self.W2(g.ndata['node'] * g.ndata['N_h']))                       # (n_users + n_entities, out_dim)
            out = out1 + out2
        else:
            raise NotImplementedError

        out = self.message_dropout(out)
        return out


class KGAT(nn.Module):
    '''
    知识图谱注意力网络(仅保留知识图谱相关部分)
    主要功能：
    1. 知识图谱实体和关系的嵌入表示
    2. 基于注意力机制的知识图谱传播
    3. TransR风格的知识图谱损失计算
    '''

    def __init__(self, args, n_entities, n_relations):
        super(KGAT, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations

        # 维度设置
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        # 网络结构参数
        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.entity_dim] + eval(args.conv_dim_list)  # 各层维度
        self.mess_dropout = eval(args.mess_dropout)  # 各层dropout率
        self.n_layers = len(eval(args.conv_dim_list))  # 层数
        self.kg_l2loss_lambda = args.kg_l2loss_lambda  # L2正则系数

        # 嵌入层
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.entity_embed = nn.Embedding(self.n_entities, self.entity_dim)

        # 关系特定的变换矩阵
        self.W_R = nn.Parameter(torch.Tensor(self.n_relations, self.entity_dim, self.relation_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        # 知识图谱聚合层
        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(
                Aggregator(self.conv_dim_list[k],
                           self.conv_dim_list[k + 1],
                           self.mess_dropout[k],
                           self.aggregation_type))

    def att_score(self, edges):
        '''
        计算知识图谱边上的注意力得分
        公式: att = (hW_r)ᵀ * tanh(tW_r + r)
        '''
        # 源节点和目标节点的变换表示
        r_mul_h = torch.matmul(self.entity_embed(edges.src['id']), self.W_r)  # (n_edge, relation_dim)
        r_mul_t = torch.matmul(self.entity_embed(edges.dst['id']), self.W_r)  # (n_edge, relation_dim)

        # 关系嵌入
        r_embed = self.relation_embed(edges.data['type'])  # (1, relation_dim)

        # 计算注意力得分
        att = torch.bmm(r_mul_h.unsqueeze(1),
                        torch.tanh(r_mul_t + r_embed).unsqueeze(2)).squeeze(-1)
        return {'att': att}

    def compute_attention(self, g):
        '''
        计算全图的注意力权重
        '''
        g = g.local_var()
        for i in range(self.n_relations):
            edge_idxs = g.filter_edges(lambda edge: edge.data['type'] == i)
            self.W_r = self.W_R[i]  # 获取当前关系的变换矩阵
            g.apply_edges(self.att_score, edge_idxs)

        # 注意力归一化
        g.edata['att'] = edge_softmax_fix(g, g.edata.pop('att'))
        return g.edata.pop('att')

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        '''
        TransR风格的知识图谱损失计算
        公式: L = -log σ(||hW_r + r - t'W_r||² - ||hW_r + r - tW_r||²)
        '''
        # 获取嵌入表示
        r_embed = self.relation_embed(r)  # (batch, relation_dim)
        W_r = self.W_R[r]  # (batch, entity_dim, relation_dim)

        h_embed = self.entity_embed(h)  # (batch, entity_dim)
        pos_t_embed = self.entity_embed(pos_t)  # (batch, entity_dim)
        neg_t_embed = self.entity_embed(neg_t)  # (batch, entity_dim)

        # 实体投影到关系空间
        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (batch, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)

        # 计算正负样本得分
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)

        # 计算损失
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        # L2正则项
        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + \
                  _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)

        return kg_loss + self.kg_l2loss_lambda * l2_loss

    def forward(self, mode, *input):
        '''
        前向传播路由
        模式:
        - 'calc_att': 计算注意力权重
        - 'calc_kg_loss': 计算知识图谱损失
        '''
        if mode == 'calc_att':
            return self.compute_attention(*input)
        if mode == 'calc_kg_loss':
            return self.calc_kg_loss(*input)
        raise ValueError(f"Unknown mode: {mode}")