import random

import torch
from torch.utils.data import Dataset
from typing import List, Tuple


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------
# Data loading
# ---------------------
def read_triples(path: str) -> List[Tuple[str, str, str]]:
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                parts = line.split(' ')
            if len(parts) != 3:
                raise ValueError(f"Bad triple line: {line}")
            h, r, t = parts
            triples.append((h, r, t))
    return triples

class KGIndex:
    def __init__(self, train, valid, test):
        ents = set()
        rels = set()
        for split in (train, valid, test):
            for h, r, t in split:
                ents.add(h); ents.add(t); rels.add(r)
        self.ent2id = {e:i for i,e in enumerate(sorted(ents))}
        self.rel2id = {r:i for i,r in enumerate(sorted(rels))}
        self.id2ent = {i:e for e,i in self.ent2id.items()}
        self.id2rel = {i:r for r,i in self.rel2id.items()}

        # 为 eval 构建过滤集
        self.hr2t = {}
        self.tr2h = {}
        for h,r,t in train + valid + test:
            hi = self.ent2id[h]; ri = self.rel2id[r]; ti = self.ent2id[t]
            self.hr2t.setdefault((hi,ri), set()).add(ti)
            self.tr2h.setdefault((ti,ri), set()).add(hi)

class TripleDataset(Dataset):
    def __init__(self, triples: List[Tuple[str,str,str]], index: KGIndex):
        self.data = [(index.ent2id[h], index.rel2id[r], index.ent2id[t]) for h,r,t in triples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
