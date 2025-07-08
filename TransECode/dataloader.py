# -*- codeing = utf-8 -*-
# @Time : 2025/7/7 14:46
# @Author : Luo_CW
# @File : dataloader.py
# @Software : PyCharm

from collections import Counter

import torch
from torch.utils import data
from typing import Dict, Tuple
from torch.utils.data import DataLoader

Mapping = Dict[str, int]

def create_mappings(dataset_path: str) -> Tuple[Mapping, Mapping]:
    """
    Creates separate mappings to indices for entities and relations.
    为实体和关系创建单独的索引映射。
    :param dataset_path: 数据存储路径
    :return:
    """
    # 计数器用于将实体/关系按最常见顺序排序
    entity_counter = Counter()
    relation_counter = Counter()
    with open(dataset_path, "r") as f:
        for line in f:
            # -1 用于删除换行符
            head, relation, tail = line[:-1].split("\t")
            entity_counter.update([head, tail])
            relation_counter.update([relation])
    entity2id = {}
    relation2id = {}
    for idx, (mid, _) in enumerate(entity_counter.most_common()):
        entity2id[mid] = idx
    for idx, (relation, _) in enumerate(relation_counter.most_common()):
        relation2id[relation] = idx
    return entity2id, relation2id

# def get_batch(triples,batch_size=50):
#     indices = torch.randperm(len(triples))[:batch_size]
#     return triples[indices]


class FB15KDataset(data.Dataset):
    """用于处理 FB15K 和 FB15K-237 的数据集实现"""

    def __init__(self, data_path: str, entity2id: Mapping, relation2id: Mapping):
        self.entity2id = entity2id
        self.relation2id = relation2id
        with open(data_path, "r") as f:
            # 数据以元组 (head, relationship, tail) 的形式表示
            self.data = [line[:-1].split("\t") for line in f]
            # print(len(self.data))
            # print(get_batch(self.data))
        self.triples = self.triples_2_id()

    def __len__(self):
        """表示样本总数"""
        return len(self.data)

    def __getitem__(self, index):
        """返回（头 ID、关系 ID、尾 ID）。"""
        head, relation, tail = self.data[index]
        head_id = self._to_idx(head, self.entity2id)
        relation_id = self._to_idx(relation, self.relation2id)
        tail_id = self._to_idx(tail, self.entity2id)
        return head_id, relation_id, tail_id

    @staticmethod
    def _to_idx(key: str, mapping: Mapping) -> int:
        '''
        其功能是将一个字符串键（key）转换为对应的整型索引（int），基于给定的映射关系（mapping）。
        根据输入的 key 从 mapping 字典中查找对应的值（索引）。
        如果 key 不存在，则返回当前字典的长度（即自动扩展映射）。
        :param key:
        :param mapping:
        :return:
        '''
        try:
            return mapping[key]
        except KeyError:
            return len(mapping)

    def triples_2_id(self):
        triples = []
        for index in range(len(self.data)):
            head, relation, tail = self.data[index]
            head_id = self._to_idx(head, self.entity2id)
            relation_id = self._to_idx(relation, self.relation2id)
            tail_id = self._to_idx(tail, self.entity2id)
            triples.append((head_id,relation_id,tail_id))
        print(len(triples))
        return triples








if __name__ == '__main__':
    train_path = '../Dataset/FB15k237/test.txt'
    entity2id, relation2id = create_mappings(train_path)
    train_set = FB15KDataset(train_path, entity2id, relation2id)
    entity2id_count = len(entity2id)
    relation2id_count = len(relation2id)
    train_dataloader_data = DataLoader(
        train_set,
        batch_size=1024,
        shuffle=True,
        num_workers=4
    )
    # train_generator = DataLoader(train_set, batch_size=128)
    for index in train_dataloader_data:
        print(index)


