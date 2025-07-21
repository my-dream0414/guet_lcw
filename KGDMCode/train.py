# -*- codeing = utf-8 -*-
# @Time : 2025/7/21 15:32
# @Author : Luo_CW
# @File : train.py
# @Software : PyCharm
from model import KGDM
import torch
from Code.DataPreprocess import KGData
from Code.setup_parser import SetupParser
import torch.nn.functional as F


def main(args):
    kgdata = KGData(args)
    num_entities = args.num_ent
    num_relations = args.num_rel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KGDM(num_entities, num_relations, emb_dim=200, timesteps=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataloader = DataLoader(

    )


    for epoch in range(args.max_steps):
        for h, r, t in dataloader:
            t_step = torch.randint(1, model.timesteps, (h.size(0),)).to(device)

            loss_denoise, pos_score = model(h.to(device), r.to(device), t.to(device), t_step)

            # margin ranking loss（负采样可以自行实现）
            neg_t = torch.randint(0, num_entities, t.size()).to(device)
            tail_neg = model.entity_emb(neg_t)
            h_emb = model.entity_emb(h)
            r_emb = model.relation_emb(r)
            r_t, _ = model.forward_diffusion(r_emb, t_step)
            score_neg = model.score(h_emb, r_t, t_step, tail_neg)

            loss_rank = F.margin_ranking_loss(pos_score, score_neg, torch.ones_like(pos_score), margin=1.0)

            loss = loss_denoise + loss_rank

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    parser = SetupParser()  # 设置参数
    args = parser.parse_args()
    main(args)