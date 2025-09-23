# ---------------------
# Training & Evaluation
# ---------------------
import random

from KGDM import KGDM
from dataLoader import KGIndex, TripleDataset,read_triples
from Diffusion import DiffusionSchedule
import argparse
import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------
# Utility: set seed
# ---------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, schedule, optimizer, device, num_negs):
    model.train()
    total = 0.0
    for batch in loader:
        batch = torch.stack(batch, dim=1).to(device)
        loss = model.forward_training_step(batch, schedule, num_negs=num_negs, device=device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)

@torch.no_grad()
def evaluate(model: KGDM, triples: List[Tuple[int,int,int]], index: KGIndex, schedule: DiffusionSchedule,
             batch_size: int, device: torch.device, hits_k=(1,3,10)):
    model.eval()
    # 在投影空间中预先计算实体库
    all_ents = torch.arange(len(index.ent2id), device=device)
    entity_bank = model.EMBe(model.ent_emb(all_ents))  # (N,D)

    def eval_mode(predict_head: bool):
        total_mrr = 0.0
        total_hits = {k: 0 for k in hits_k}
        count = 0
        for i in range(0, len(triples), batch_size):
            batch = triples[i:i+batch_size]
            if predict_head:
                # (?, r, t)
                h_idx = torch.tensor([h for (h,_,_) in batch], device=device)
                r_idx = torch.tensor([r for (_,r,_) in batch], device=device)
                t_idx = torch.tensor([t for (_,_,t) in batch], device=device)
                # 交换角色：尾部条件和关系通过对称性生成头部
                # 我们做了一个简单的 hack：如果 r_inv 存在，则处理 (t, r_inv, h)；否则重复使用 r
                # 为了忠实地再现，可以添加一个专用的 head 生成器；这里我们镜像 tail gen。
                # 我们只需以 (t, r) 作为条件来生成 head。
                ranks_lists, dists = model.rank_tail(t_idx, r_idx, schedule, entity_bank, index.tr2h, hits_k)
                golds = h_idx
                filtered_map = index.tr2h
                cond_pairs = list(zip(t_idx.tolist(), r_idx.tolist()))
            else:
                # (h, r, ?)
                h_idx = torch.tensor([h for (h,_,_) in batch], device=device)
                r_idx = torch.tensor([r for (_,r,_) in batch], device=device)
                t_idx = torch.tensor([t for (_,_,t) in batch], device=device)
                ranks_lists, dists = model.rank_tail(h_idx, r_idx, schedule, entity_bank, index.hr2t, hits_k)
                golds = t_idx
                filtered_map = index.hr2t
                cond_pairs = list(zip(h_idx.tolist(), r_idx.tolist()))

            for j, order in enumerate(ranks_lists):
                gold = golds[j].item()
                pair = cond_pairs[j]
                # 过滤设置：删除除当前黄金之外的所有有效实体
                filtered_set = filtered_map.get(pair, set())
                filtered_list = [e for e in order.tolist() if (e not in filtered_set) or (e == gold)]
                rank = filtered_list.index(gold) + 1
                total_mrr += 1.0 / rank
                for k in hits_k:
                    total_hits[k] += 1 if rank <= k else 0
                count += 1
        return total_mrr / count, {k: v / count for k,v in total_hits.items()}

    mrr_tail, hits_tail = eval_mode(False)
    mrr_head, hits_head = eval_mode(True)
    # 按照通常做法平均
    mrr = (mrr_tail + mrr_head) / 2.0
    hits = {k: (hits_tail[k] + hits_head[k]) / 2.0 for k in hits_k}
    return mrr, hits

# ---------------------
# Main
# ---------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./FB15k-237', help='Directory with train.txt, valid.txt, test.txt')
    parser.add_argument('--dataset', type=str, default='FB15k-237')
    parser.add_argument('--emb_dim', type=int, default=400)  # base embedding dim
    parser.add_argument('--proj_dim', type=int, default=1000)  # projected working dim (paper uses large dims)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--hidden', type=int, default=1200)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--num_negs', type=int, default=64)
    parser.add_argument('--scoring', type=str, default='add', choices=['add','hadamard'],
                        help='Use add (FB15k-237) or hadamard (WN18RR/Kinship/UMLS)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    set_seed(42)
    print("开始加载数据")
    train_triples = read_triples(os.path.join(args.data_dir, 'train.txt'))
    valid_triples = read_triples(os.path.join(args.data_dir, 'valid.txt'))
    test_triples  = read_triples(os.path.join(args.data_dir, 'test.txt'))

    index = KGIndex(train_triples, valid_triples, test_triples)

    # 映射到 ID
    train_ids = [(index.ent2id[h], index.rel2id[r], index.ent2id[t]) for (h,r,t) in train_triples]
    valid_ids = [(index.ent2id[h], index.rel2id[r], index.ent2id[t]) for (h,r,t) in valid_triples]
    test_ids  = [(index.ent2id[h], index.rel2id[r], index.ent2id[t]) for (h,r,t) in test_triples]
    print("数据加载完成，开始处理数据！")
    train_ds = TripleDataset(train_triples, index)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              collate_fn=lambda b: [torch.tensor(x) for x in zip(*b)])

    device = torch.device(args.device)
    print("数据处理完成，开始搭建模型！")
    model = KGDM(num_entities=len(index.ent2id), num_relations=len(index.rel2id),
                 in_dim=args.emb_dim, proj_dim=args.proj_dim, T=args.num_steps,
                 scoring=args.scoring, hidden=args.hidden, num_blocks=args.num_blocks,
                 gamma=args.gamma).to(device)

    schedule = DiffusionSchedule.create(T=args.num_steps, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print("模型搭建完成，开始训练模型！")
    for epoch in range(1, args.epochs+1):
        loss = train_one_epoch(model, train_loader, schedule, optimizer, device, num_negs=args.num_negs)
        print(f"Epoch {epoch}: loss={loss:.4f}")

        # 快速验证
        val_mrr, val_hits = evaluate(model,
                                     [(h,r,t) for (h,r,t) in valid_ids],
                                     index, schedule, batch_size=256, device=device)
        print(f"  Valid MRR={val_mrr:.4f} | Hits@1={val_hits[1]:.4f} Hits@3={val_hits[3]:.4f} Hits@10={val_hits[10]:.4f}")

    # 最终测试
    test_mrr, test_hits = evaluate(model,
                                   [(h,r,t) for (h,r,t) in test_ids],
                                   index, schedule, batch_size=256, device=device)
    print(f"TEST  MRR={test_mrr:.4f} | Hits@1={test_hits[1]:.4f} Hits@3={test_hits[3]:.4f} Hits@10={test_hits[10]:.4f}")

if __name__ == '__main__':
    main()
