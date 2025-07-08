import numpy as np
from torch.utils.data import Dataset
import torch
import os
from collections import defaultdict as ddict
from IPython import embed

class KGData(object):
    """
    kg 数据预处理。
    属性：
        args：一些预设参数，例如数据集路径等。
        ent2id：将实体编码为三元组，类型：dict。
        rel2id：将关系编码为三元组，类型：dict。
        id2ent：将实体解码为三元组，类型：dict。
        id2rel：将关系解码为三元组，类型：dict。
        train_triples：记录用于训练的三元组，类型：list。
        valid_triples：记录用于验证的三元组，类型：list。
        test_triples：记录用于测试的三元组，类型：list。
        all_true_triples：记录所有三元组，包括训练集、验证集和测试集，类型：list。
        TrainTriples
        Relation2Tuple
        RelSub2Obj
        hr2t_train：记录相同头和关系对应的尾，类型：defaultdict(class:set)。
        rt2h_train：记录同一个tail对应的head，类型为defaultdict(class:set)。
        h2rt_train：记录同一个head对应的tail，类型为defaultdict(class:set)。
        t2rh_train：记录同一个tail对应的head，类型为defaultdict(class:set)。
    """
    # TODO:把里面的函数再分一分，最基础的部分再初始化的使用调用，其他函数具体情况再调用
    def __init__(self, args):
        self.args = args

        # 基础部分
        self.ent2id = {}
        self.rel2id = {}
        # 预测器需要
        self.id2ent = {}
        self.id2rel = {}
        # 存放三元组的id
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        self.all_true_triples = set()

        #  grounding 使用
        self.TrainTriples = {}
        self.Relation2Tuple = {}
        self.RelSub2Obj = {}

        self.hr2t_train = ddict(set)
        self.rt2h_train = ddict(set)
        self.h2rt_train = ddict(set)
        self.t2rh_train = ddict(set)
        self.get_id()
        self.get_triples_id()
        if args.use_weight:
            self.count = self.count_frequency(self.train_triples)

    def get_id(self):
        """
        获取实体/关系 ID 以及实体/关系编号。
        获取实体总数与关系总数
        更新：
        self.ent2id：实体到 ID。
        self.rel2id：关系到 ID。
        self.id2ent：ID 到实体。
        self.id2rel：ID 到关系。
        self.args.num_ent：实体编号。
        self.args.num_rel：关系编号。
        """
        with open(os.path.join(self.args.data_path, "entities.dict"),encoding='utf-8') as fin:
            for line in fin:
                eid, entity = line.strip().split("\t")
                self.ent2id[entity] = int(eid)
                self.id2ent[int(eid)] = entity

        with open(os.path.join(self.args.data_path, "relations.dict"),encoding='utf-8') as fin:
            for line in fin:
                rid, relation = line.strip().split("\t")
                self.rel2id[relation] = int(rid)
                self.id2rel[int(rid)] = relation

        self.args.num_ent = len(self.ent2id)
        self.args.num_rel = len(self.rel2id)


    def get_triples_id(self):
        """
        获取三元组 ID，并以 (h, r, t) 格式保存。
        更新：
            self.train_triples：训练数据集三元组 ID。
            self.valid_triples：有效数据集三元组 ID。
            self.test_triples：测试数据集三元组 ID。
        """
        
        with open(os.path.join(self.args.data_path, "train.txt"),encoding='utf-8') as f:

            for line in f.readlines():
                h, r, t = line.strip().split()
                self.train_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )
                
                tmp = str(self.ent2id[h]) + '\t' + str(self.rel2id[r]) + '\t' + str(self.ent2id[t])
                self.TrainTriples[tmp] = True

                iRelationID = self.rel2id[r]
                strValue = str(h) + "#" + str(t)
                if not iRelationID in self.Relation2Tuple:
                    tmpLst = []
                    tmpLst.append(strValue)
                    self.Relation2Tuple[iRelationID] = tmpLst
                else:
                    self.Relation2Tuple[iRelationID].append(strValue)

                iRelationID = self.rel2id[r]
                iSubjectID = self.ent2id[h]
                iObjectID = self.ent2id[t]
                tmpMap = {}
                tmpMap_in = {}
                if not iRelationID in self.RelSub2Obj:
                    if not iSubjectID in tmpMap:
                        tmpMap_in.clear()
                        tmpMap_in[iObjectID] = True
                        tmpMap[iSubjectID] = tmpMap_in
                    else:
                        tmpMap[iSubjectID][iObjectID] = True
                    self.RelSub2Obj[iRelationID] = tmpMap
                else:
                    tmpMap = self.RelSub2Obj[iRelationID]
                    if not iSubjectID in tmpMap:
                        tmpMap_in.clear()
                        tmpMap_in[iObjectID] = True
                        tmpMap[iSubjectID] = tmpMap_in
                    else:
                        tmpMap[iSubjectID][iObjectID] = True
                    self.RelSub2Obj[iRelationID] = tmpMap  # 是不是应该要加？

        with open(os.path.join(self.args.data_path, "valid.txt"),encoding='utf-8') as f:

            for line in f.readlines():
                h, r, t = line.strip().split()
                self.valid_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )


        with open(os.path.join(self.args.data_path, "test.txt"),encoding='utf-8') as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.test_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        self.all_true_triples = set(
            self.train_triples + self.valid_triples + self.test_triples
        )

    def get_hr2t_rt2h_from_train(self):
        """
        从训练数据集中获取 hr2t 和 rt2h 的集合，数据类型为 numpy。
        更新：
            self.hr2t_train：hr2t 的集合。
            self.rt2h_train：rt2h 的集合。
        """
        for h, r, t in self.train_triples:
            self.hr2t_train[(h, r)].add(t)
            self.rt2h_train[(r, t)].add(h)
        for h, r in self.hr2t_train:
            self.hr2t_train[(h, r)] = np.array(list(self.hr2t_train[(h, r)]))
        for r, t in self.rt2h_train:
            self.rt2h_train[(r, t)] = np.array(list(self.rt2h_train[(r, t)]))


    @staticmethod
    def count_frequency(triples, start=4):
        '''
        获取部分三元组（例如 (head, relationship) 或 (relation, tail)）的频率。
        该频率将用于类似 word2vec 的子采样。
        参数：
            triples：采样的三元组。
            start：初始计数。
        返回：
            count：记录 (head, relationship) 的数量。
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
        

    def get_h2rt_t2hr_from_train(self):
        """
        从训练数据集中获取 h2rt 和 t2hr 的集合，数据类型为 numpy。

        更新：
            self.h2rt_train：h2rt 的集合。
            self.t2rh_train：t2hr 的集合。
        """
        for h, r, t in self.train_triples:
            self.h2rt_train[h].add((r, t))
            self.t2rh_train[t].add((r, h))
        for h in self.h2rt_train:
            self.h2rt_train[h] = np.array(list(self.h2rt_train[h]))
        for t in self.t2rh_train:
            self.t2rh_train[t] = np.array(list(self.t2rh_train[t]))
        
    def get_hr_trian(self):
        '''
        更改批次的生成模式。
            合并具有相同头和关系的三元组，以进行 1vsN 训练模式。
        返回：
            self.train_triples：用于训练的 tuple(hr, t) 列表
        '''
        self.t_triples = self.train_triples 
        self.train_triples = [ (hr, list(t)) for (hr,t) in self.hr2t_train.items()]

class BaseSampler(KGData):
    """
    传统的随机抽样模式
    """
    def __init__(self, args):
        super().__init__(args)
        self.get_hr2t_rt2h_from_train()

    def corrupt_head(self, t, r, num_max=1):
        """
        对头部实体进行负采样。
        参数：
            t：三元组中的尾部实体。
            r：三元组中的关系。
            num_max：生成的负样本的最大值。
        返回：
            neg：过滤掉正头部实体的头部实体的负样本。
        """
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        # 在调用 in1d 前，先确保它是 ndarray：
        candidates = self.rt2h_train.get((r, t), [])
        if not isinstance(candidates, np.ndarray):
            candidates = np.array(candidates, dtype=np.int64)
        tmp = np.array(tmp, dtype=np.int64)  # 假设你是处理实体 ID 或关系 ID

        mask = np.in1d(tmp, candidates, assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def corrupt_tail(self, h, r, num_max=1):
        """
        尾部实体的负采样。
        参数：
            h：三元组中的头实体。
            r：三元组中的关系。
            num_max：生成的负样本的最大值。
        返回：
            neg：过滤掉正尾实体的尾部实体的负样本。
        """
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp

        # 在调用 in1d 前，先确保它是 ndarray：
        candidates = self.hr2t_train.get((h, r), [])
        if not isinstance(candidates, np.ndarray):
            candidates = np.array(candidates, dtype=np.int64)
        tmp = np.array(tmp, dtype=np.int64)  # 假设你是处理实体 ID 或关系 ID

        mask = np.in1d(tmp, candidates, assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def head_batch(self, h, r, t, neg_size=None):
        """
        头部实体的负采样。
        参数：
            h：三元组中的头部实体
            t：三元组中的尾部实体。
            r：三元组中的关系。
            neg_size：负样本的大小。
        返回：
            头部实体的负样本。[neg_size]
        """
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_head(t, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def tail_batch(self, h, r, t, neg_size=None):
        """
        尾部实体的负采样。
        参数：
            h：三元组中的头实体
            t：三元组中的尾部实体。
            r：三元组中的关系。
            neg_size：负样本的大小。
        返回：
            尾部实体的负样本。[neg_size]
        """
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_tail(h, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def get_train(self):
        return self.train_triples
    def get_valid(self):
        return self.valid_triples
    def get_test(self):
        return self.test_triples
    def get_all_true_triples(self):
        return self.all_true_triples


class RevSampler(KGData):
    """
    在传统随机采样模式下添加反向三元组。

    对于每个三元组 (h, r, t)，生成反向三元组 (t, r`, h)。
    r` = r + num_rel。

    属性：
        hr2t_train：记录相同头部和关系对应的尾部，类型：defaultdict(class:set)。
        rt2h_train：记录相同尾部和关系对应的头部，类型：defaultdict(class:set)。
    """
    def __init__(self, args):
        super().__init__(args)
        self.hr2t_train = ddict(set)
        self.rt2h_train = ddict(set)
        self.add_reverse_relation()
        self.add_reverse_triples()
        self.get_hr2t_rt2h_from_train()

    def add_reverse_relation(self):
        """
        获取实体/关系/反向关系 ID 以及实体/关系编号。
        更新：
            self.ent2id：实体 ID。
            self.rel2id：关系 ID。
            self.args.num_ent：实体编号。
            self.args.num_rel：关系编号。
        """
        with open(os.path.join(self.args.data_path, "relations.dict")) as fin:
            len_rel2id = len(self.rel2id)
            for line in fin:
                rid, relation = line.strip().split("\t")
                self.rel2id[relation + "_reverse"] = int(rid) + len_rel2id
                self.id2rel[int(rid) + len_rel2id] = relation + "_reverse"
        self.args.num_rel = len(self.rel2id)

    def add_reverse_triples(self):
        """
        生成反向三元组 (t, r`, h)。
        更新：
            self.train_triples：用于训练的三元组。
            self.valid_triples：用于验证的三元组。
            self.test_triples：用于测试的三元组。
            self.all_ture_triples：包括训练集、验证集和测试集在内的所有三元组。
        """
        with open(os.path.join(self.args.data_path, "train.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.train_triples.append(
                    (self.ent2id[t], self.rel2id[r + "_reverse"], self.ent2id[h])
                )

        with open(os.path.join(self.args.data_path, "valid.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.valid_triples.append(
                    (self.ent2id[t], self.rel2id[r + "_reverse"], self.ent2id[h])
                )

        with open(os.path.join(self.args.data_path, "test.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.test_triples.append(
                    (self.ent2id[t], self.rel2id[r + "_reverse"], self.ent2id[h])
                )

        self.all_true_triples = set(
            self.train_triples + self.valid_triples + self.test_triples
        )

    def get_train(self):
        return self.train_triples

    def get_valid(self):
        return self.valid_triples

    def get_test(self):
        return self.test_triples

    def get_all_true_triples(self):
        return self.all_true_triples    
    
    def corrupt_head(self, t, r, num_max=1):
        """
        对头部实体进行负采样。

        参数：
            t：三元组中的尾部实体。
            r：三元组中的关系。
            num_max：生成的负样本的最大值。

        返回：
            neg：过滤掉正头部实体的头部实体的负样本。
        """
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp

        # 在调用 in1d 前，先确保它是 ndarray：
        candidates = self.rt2h_train.get((r, t), [])
        if not isinstance(candidates, np.ndarray):
            candidates = np.array(candidates, dtype=np.int64)
        tmp = np.array(tmp, dtype=np.int64)  # 假设你是处理实体 ID 或关系 ID

        mask = np.in1d(tmp, candidates, assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def corrupt_tail(self, h, r, num_max=1):
        """
        尾部实体的负采样。

        参数：
            h：三元组中的头实体。
            r：三元组中的关系。
            num_max：生成的负样本的最大值。

        返回：
            neg：过滤掉正尾实体的尾部实体的负样本。
        """
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp

        # 在调用 in1d 前，先确保它是 ndarray：
        candidates = self.hr2t_train.get((h, r), [])
        if not isinstance(candidates, np.ndarray):
            candidates = np.array(candidates, dtype=np.int64)
        tmp = np.array(tmp, dtype=np.int64)  # 假设你是处理实体 ID 或关系 ID

        mask = np.in1d(tmp, candidates, assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def head_batch(self, h, r, t, neg_size=None):
        """
        头部实体的负采样。
        参数：
            h：三元组中的头部实体
            t：三元组中的尾部实体。
            r：三元组中的关系。
            neg_size：负样本的大小。
        返回：
            头部实体的负样本。[neg_size]
        """
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_head(t, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def tail_batch(self, h, r, t, neg_size=None):
        """
        尾部实体的负采样。
        参数：
        h：三元组中的头实体
        t：三元组中的尾部实体。
        r：三元组中的关系。
        neg_size：负样本的大小。
        返回：
            尾部实体的负样本。[neg_size]
        """
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_tail(h, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]