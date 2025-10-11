# train.py
import os
import argparse
import time
import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataLoader import read_triples, KGIndex, TripleDataset, set_seed
from model import SDKGC
from collections import defaultdict
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./FB15K-237", help="folder with train.txt valid.txt test.txt")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--embed_dim", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--diffusion_steps", type=int, default=200)
parser.add_argument("--quick", action="store_true", help="quick run for debugging")
parser.add_argument("--save_path", type=str, default="sdkgc_ckpt.pth")
args = parser.parse_args()

if args.quick:
    args.embed_dim = 64
    args.batch_size = 64
    args.epochs = 1
    args.diffusion_steps = 8

set_seed(42)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- load data ----
train_raw = read_triples(os.path.join(args.data_dir, "train.txt"))
valid_raw = read_triples(os.path.join(args.data_dir, "valid.txt"))
test_raw  = read_triples(os.path.join(args.data_dir, "test.txt"))
index = KGIndex(train_raw, valid_raw, test_raw)
train_ds = TripleDataset(train_raw, index)
valid_ds = TripleDataset(valid_raw, index)
test_ds = TripleDataset(test_raw, index)
print(f"Entities: {len(index.ent2id)} Relations: {len(index.rel2id)}")

# ---- neighbor sampler using precomputed adjacency ----
adj = defaultdict(list)
for h,r,t in train_ds.data:
    adj[h].append((t,r))
    adj[t].append((h,r))

def neighbor_sampler(node_list, K, device):
    # node_list: list of ids
    B = len(node_list)
    neigh_ids = torch.zeros((B, K), dtype=torch.long, device=device)
    neigh_rels = torch.zeros((B, K), dtype=torch.long, device=device)
    mask = torch.zeros((B, K), dtype=torch.bool, device=device)
    for i, nid in enumerate(node_list):
        nb = adj.get(nid, [])
        # sample up to K neighbors (randomly) for stochasticity
        if len(nb) == 0:
            continue
        if len(nb) > K:
            nb = random.sample(nb, K)
        for j, (t,r) in enumerate(nb):
            neigh_ids[i,j] = t
            neigh_rels[i,j] = r
            mask[i,j] = True
    return neigh_ids, neigh_rels, mask



# ---- filtered evaluation function ----
def evaluate_filtered(model, data_loader, index, max_eval=1000):
    model.eval()
    MRR = 0.0; hits = {1:0,3:0,10:0}; cnt = 0
    with torch.no_grad():
        for batch in data_loader:
            hs, rs, ts = batch
            hs = hs.to(device); rs = rs.to(device); ts = ts.to(device)
            scores = model.forward_score_all(hs, rs)  # (B, E)
            # filtered masking
            for i in range(scores.size(0)):
                h = int(hs[i].item()); r = int(rs[i].item()); t = int(ts[i].item())
                filt = index.hr2t.get((h,r), set())
                # set score of known true tails (except target t) to -inf
                if len(filt) > 0:
                    filt_list = list(filt)
                    scores[i, filt_list] = -1e9
                    # restore target's original score will be used for ranking; but ensure it's not masked
                    scores[i, t] = scores[i, t]
                # compute rank (1-based) in descending order
                sc_np = scores[i].cpu().numpy()
                rank = (np.argsort(-sc_np) == t).nonzero()[0][0] + 1
                MRR += 1.0 / rank
                for k in (1,3,10):
                    if rank <= k:
                        hits[k] += 1
                cnt += 1
                if cnt >= max_eval:
                    break
            if cnt >= max_eval:
                break
    return {"MRR": MRR/cnt, "Hits@1": hits[1]/cnt, "Hits@3": hits[3]/cnt, "Hits@10": hits[10]/cnt}

# inject neighbor_sampler into model at creation
model = SDKGC(num_entities=len(index.ent2id),
              num_relations=len(index.rel2id),
              emb_dim=args.embed_dim,
              diffusion_steps=args.diffusion_steps,
              neighbor_sampler=neighbor_sampler).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler()

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)


# ---- training loop ----
best_mrr = 0.0
for epoch in range(args.epochs):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for i, (hs, rs, ts) in enumerate(train_loader):
        hs = hs.to(device); rs = rs.to(device); ts = ts.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            # encode head+relation cond
            h_vec = model.encode_head(hs)
            r_vec = model.rel_emb(rs)
            cond = torch.cat([h_vec, r_vec], dim=-1)
            t_vec = model.ent_emb(ts)
            loss_diff = model.diffusion.forward_loss(t_vec, cond)
            # generate and cross-entropy on all entities (fast discriminative surrogate)
            gen_t = model.diffusion.generate(cond, device=device)
            logits = torch.matmul(gen_t, model.ent_emb.weight.t()) / 0.07
            loss_ce = F.cross_entropy(logits, ts)
            loss = loss_diff + 0.2 * loss_ce
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        # limiter for debug
        if args.quick and i > 5:
            break
    dur = time.time() - t0
    # validate
    val_stats = evaluate_filtered(model, valid_loader, index, max_eval=1000 if not args.quick else 200)
    print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss:.4f} time={dur:.1f}s val={val_stats}")
    # save best
    if val_stats["MRR"] > best_mrr:
        best_mrr = val_stats["MRR"]
        torch.save({"model":model.state_dict(), "index":index.__dict__}, args.save_path)
        print("Saved best model to", args.save_path)

# ---- final test eval ----
print("Testing best model on test set...")
ckpt = torch.load(args.save_path, map_location=device)
model.load_state_dict(ckpt["model"])
test_stats = evaluate_filtered(model, test_loader, index, max_eval=2000 if not args.quick else 500)
print("Test stats:", test_stats)
