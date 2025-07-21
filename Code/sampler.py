# -*- codeing = utf-8 -*-
# @Time : 2025/7/8 16:13
# @Author : Luo_CW
# @File : sampler.py
# @Software : PyCharm
from numpy.random.mtrand import normal
import random
from DataPreprocess import *
import dgl
import time
import queue
import math


class UniSampler(BaseSampler):
    """
    随机负采样
        过滤掉正样本，并随机选取部分样本作为负样本。
    属性：
        cross_sampling_flag：交叉采样头尾负样本的标志。
    """
    def __init__(self, args):
        super().__init__(args)
        self.cross_sampling_flag = 0

    def sampling(self, data):
        """
        过滤掉正样本，并随机选取一些样本作为负样本.
        参数：
            data：用于采样的三元组。
        返回：
            batch_data：训练数据。
        """
        batch_data = {}
        neg_ent_sample = []
        subsampling_weight = []
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        if self.cross_sampling_flag == 0:
            batch_data['mode'] = "head-batch"
            for h, r, t in data:
                neg_head = self.head_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_head)
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r - 1)]
                    subsampling_weight.append(weight)
        else:
            batch_data['mode'] = "tail-batch"
            for h, r, t in data:
                neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_tail)
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r - 1)]
                    subsampling_weight.append(weight)

        # 处理之前的数据 3153 11 8874
        # 处理之后的数据 [11240  8497  6557 14450  4315  5123 12985  3272 11995  4567]
        '''
        处理之前的数据 3153 11 8874
        处理之后的数据 [11240  8497  6557 14450  4315  5123 12985  3272 11995  4567]
        其中根据处理方式：如果是替换头实体：处理后的数据则全是负样本的头实体
        如果是替换尾实体：处理后的数据则全是负样本的尾实体
        '''

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_sample'] = torch.LongTensor(np.array(neg_ent_sample))
        if self.args.use_weight:
            batch_data["subsampling_weight"] = torch.sqrt(1 / torch.tensor(subsampling_weight))
        return batch_data

    def uni_sampling(self, data):
        '''
        用于对知识图谱中的三元组进行统一的负采样（uniform negative sampling）。
        为每一个正样本三元组 (h, r, t) 生成一定数量的头实体和尾实体的负样本，从而构建训练数据。
        :param data:
        :return:
        '''
        batch_data = {}
        neg_head_list = []
        neg_tail_list = []
        for h, r, t in data:
            neg_head = self.head_batch(h, r, t, self.args.num_neg)
            neg_head_list.append(neg_head)
            neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
            neg_tail_list.append(neg_tail)

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_head'] = torch.LongTensor(np.array(neg_head_list))
        batch_data['negative_tail'] = torch.LongTensor(np.array(neg_tail_list))
        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'mode']


class BernSampler(BaseSampler):
    """
    使用伯努利分布来选择是否替换头部实体或尾部实体。
    属性：
        lef_mean：记录头部实体的平均值
        rig_mean：记录尾部实体的平均值
    """
    def __init__(self, args):
        super().__init__(args)
        self.lef_mean, self.rig_mean = self.calc_bern()

    def __normal_batch(self, h, r, t, neg_size):
        """
        根据伯努利分布生成替换头/尾列表。
        参数：
            h：三元组的头。
            r：三元组的关系。
            t：三元组的尾。
            neg_size：每个三元组对应的负样本数量。
        返回：
            numpy.array：替换头列表和替换尾列表。
        """
        neg_size_h = 0
        neg_size_t = 0
        prob = self.rig_mean[r] / (self.rig_mean[r] + self.lef_mean[r])
        for i in range(neg_size):
            if random.random() > prob:
                neg_size_h += 1
            else:
                neg_size_t += 1
        res = []
        neg_list_h = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_h:
            neg_tmp_h = self.corrupt_head(t, r, num_max=(neg_size_h - neg_cur_size) * 2)
            neg_list_h.append(neg_tmp_h)
            neg_cur_size += len(neg_tmp_h)
        if neg_list_h != []:
            neg_list_h = np.concatenate(neg_list_h)

        for hh in neg_list_h[:neg_size_h]:
            res.append((hh, r, t))

        neg_list_t = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_t:
            neg_tmp_t = self.corrupt_tail(h, r, num_max=(neg_size_t - neg_cur_size) * 2)
            neg_list_t.append(neg_tmp_t)
            neg_cur_size += len(neg_tmp_t)
        if neg_list_t != []:
            neg_list_t = np.concatenate(neg_list_t)

        for tt in neg_list_t[:neg_size_t]:
            res.append((h, r, tt))

        return res

    def sampling(self, data):
        """
        使用伯努利分布来选择是否替换头实体或尾实体。
        参数：
            data：用于采样的三元组。
        返回：
            batch_data：训练数据。
        """
        batch_data = {}
        neg_ent_sample = []

        batch_data['mode'] = 'bern'
        for h, r, t in data:
            neg_ent = self.__normal_batch(h, r, t, self.args.num_neg)
            neg_ent_sample += neg_ent

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data["negative_sample"] = torch.LongTensor(np.array(neg_ent_sample))

        return batch_data

    def calc_bern(self):
        """
        计算 lef_mean 和 rig_mean。
        返回：
            lef_mean：记录头部实体的平均值。
            rig_mean：记录尾部实体的平均值。
        """
        h_of_r = ddict(set)
        t_of_r = ddict(set)
        freqRel = ddict(float)
        lef_mean = ddict(float)
        rig_mean = ddict(float)
        for h, r, t in self.train_triples:
            freqRel[r] += 1.0
            h_of_r[r].add(h)
            t_of_r[r].add(t)
        for r in h_of_r:
            lef_mean[r] = freqRel[r] / len(h_of_r[r])
            rig_mean[r] = freqRel[r] / len(t_of_r[r])
        return lef_mean, rig_mean

    @staticmethod
    def sampling_keys():
        return ['positive_sample', 'negative_sample', 'mode']


class AdvSampler(BaseSampler):
    """自我对抗负采样，数学表达式：
    p\left(h_{j}^{\prime}, r, t_{j}^{\prime} \mid\left\{\left(h_{i}, r_{i}, t_{i}\right)\right\}\right)=\frac{\exp \alpha f_{r}\left(\mathbf{h}_{j}^{\prime}, \mathbf{t}_{j}^{\prime}\right)}{\sum_{i} \exp \alpha f_{r}\left(\mathbf{h}_{i}^{\prime}, \mathbf{t}_{i}^{\prime}\right)}
    属性：
        freq_hr：(h, r) 对的数量。
        freq_tr：(t, r) 对的数量。
    """

    def __init__(self, args):
        super().__init__(args)
        self.freq_hr, self.freq_tr = self.calc_freq()

    def sampling(self, pos_sample):
        """
        自我对抗负采样。
        Args:
            data: 曾经对三元组进行采样。
        Returns:
            batch_data: 训练数据
        """
        data = pos_sample.numpy().tolist()
        adv_sampling = []
        for h, r, t in data:
            weight = self.freq_hr[(h, r)] + self.freq_tr[(t, r)]
            adv_sampling.append(weight)
        adv_sampling = torch.tensor(adv_sampling, dtype=torch.float32).cuda()
        adv_sampling = torch.sqrt(1 / adv_sampling)
        return adv_sampling

    def calc_freq(self):
        """Calculating the freq_hr and freq_tr.
        Returns:
            freq_hr: The count of (h, r) pairs.
            freq_tr: The count of (t, r) pairs.
        """
        freq_hr, freq_tr = {}, {}
        for h, r, t in self.train_triples:
            if (h, r) not in freq_hr:
                freq_hr[(h, r)] = self.args.freq_init
            else:
                freq_hr[(h, r)] += 1
            if (t, r) not in freq_tr:
                freq_tr[(t, r)] = self.args.freq_init
            else:
                freq_tr[(t, r)] += 1
        return freq_hr, freq_tr


class AllSampler(RevSampler):
    """Merging triples which have same head and relation, all false tail entities are taken as negative samples.
    合并具有相同头和关系的三元组，所有假尾实体均被视为负样本
    """

    def __init__(self, args):
        super().__init__(args)
        # self.num_rel_without_rev = self.args.num_rel // 2

    def sampling(self, data):
        """
        从合并后的三元组中随机抽样。
        参数：
            data：用于抽样的三元组。
        返回：
            batch_data：训练数据。
        """
        # sample_id = [] #确定triple里的relation是否是reverse的。reverse为1，不是为0
        batch_data = {}
        table = torch.zeros(len(data), self.args.num_ent)
        for id, (h, r, _) in enumerate(data):
            hr_sample = self.hr2t_train[(h, r)]
            table[id][hr_sample] = 1
            # if r > self.num_rel_without_rev:
            #     sample_id.append(1)
            # else:
            #     sample_id.append(0)
        batch_data["sample"] = torch.LongTensor(np.array(data))
        batch_data["label"] = table.float()
        # batch_data["sample_id"] = torch.LongTensor(sample_id)
        return batch_data

    def sampling_keys(self):
        return ["sample", "label"]


class CrossESampler(BaseSampler):
    # TODO:类名还需要商榷下
    def __init__(self, args):
        super().__init__(args)
        self.neg_weight = float(self.args.neg_weight / self.args.num_ent)

    def sampling(self, data):
        '''一个样本同时做head/tail prediction'''
        batch_data = {}
        hr_label = self.init_label(len(data))
        tr_label = self.init_label(len(data))
        for id, (h, r, t) in enumerate(data):
            hr_sample = self.hr2t_train[(h, r)]
            hr_label[id][hr_sample] = 1.0
            tr_sample = self.rt2h_train[(r, t)]
            tr_label[id][tr_sample] = 1.0
        batch_data["sample"] = torch.LongTensor(data)
        batch_data["hr_label"] = hr_label.float()
        batch_data["tr_label"] = tr_label.float()
        return batch_data

    def init_label(self, row):
        label = torch.rand(row, self.args.num_ent)
        label = (label > self.neg_weight).float()
        label -= 1.0
        return label

    def sampling_keys(self):
        return ["sample", "label"]


class ConvSampler(RevSampler):  # TODO:SEGNN
    """
    合并具有相同头和关系的三元组，所有假尾实体均视为负样本。
    具有相同头和关系的三元组被视为一个三元组。
    属性：
        label：将假尾实体屏蔽为负样本。
        triples：用于采样的三元组。
    """

    def __init__(self, args):
        self.label = None
        self.triples = None
        super().__init__(args)
        super().get_hr_trian()

    def sampling(self, pos_hr_t):
        """
        从合并后的三元组中随机抽样。
        参数：
            pos_hr_t：用于抽样的三元组 ((head,relation) 对)。
        返回：
            batch_data：训练数据。
        """
        batch_data = {}
        t_triples = []
        self.label = torch.zeros(self.args.train_bs, self.args.num_ent)
        self.triples = torch.LongTensor([hr for hr, _ in pos_hr_t])
        for hr, t in pos_hr_t:
            t_triples.append(t)

        for id, hr_sample in enumerate([t for _, t in pos_hr_t]):
            self.label[id][hr_sample] = 1

        batch_data["sample"] = self.triples
        batch_data["label"] = self.label
        batch_data["t_triples"] = t_triples

        return batch_data

    def sampling_keys(self):
        return ["sample", "label", "t_triples"]


class XTransESampler(RevSampler):
    """
    随机负采样并记录邻居实体。
    属性：
        triples：用于采样的三元组。
        neg_sample：负样本。
        h_neighbor：采样实体的邻居。
        h_mask：有效邻居的标签。
        max_neighbor：邻居实体的最大值。
    """

    def __init__(self, args):
        super().__init__(args)
        super().get_h2rt_t2hr_from_train()
        self.triples = None
        self.neg_sample = None
        self.h_neighbor = None
        self.h_mask = None
        self.max_neighbor = 200

    def sampling(self, data):
        """
        随机负采样并记录邻近实体。
        参数：
            data：用于采样的三元组。
        返回：
            batch_data：训练数据。
        """
        batch_data = {}

        neg_ent_sample = []
        mask = np.zeros([self.args.train_bs, 20000], dtype=float)
        h_neighbor = np.zeros([self.args.train_bs, 20000, 2])

        for id, triples in enumerate(data):
            h, r, t = triples
            num_h_neighbor = len(self.h2rt_train[h])
            h_neighbor[id][0:num_h_neighbor] = np.array(self.h2rt_train[h])

            mask[id][0:num_h_neighbor] = np.ones([num_h_neighbor])

            neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
            neg_ent_sample.append(neg_tail)

        self.triples = data
        self.neg_sample = neg_ent_sample
        self.h_neighbor = h_neighbor[:, :self.max_neighbor]
        self.h_mask = mask[:, :self.max_neighbor]

        batch_data["positive_sample"] = torch.LongTensor(self.triples)
        batch_data['negative_sample'] = torch.LongTensor(self.neg_sample)
        batch_data['neighbor'] = torch.LongTensor(self.h_neighbor)
        batch_data['mask'] = torch.LongTensor(self.h_mask)
        batch_data['mode'] = "tail-batch"
        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'neighbor', 'mask', 'mode']


class GraphSampler(RevSampler):
    """
    神经网络中基于图的采样。
    属性：
        entity：采样三元组的实体。
        relation：采样三元组之间的关系。
        triples：采样三元组。
        graph：DGL 中由 dgl.graph 构建的采样三元组图。
        norm：图中的边范数。
        label：将假尾部掩码为负样本。
    """

    def __init__(self, args):
        super().__init__(args)
        self.entity = None
        self.relation = None
        self.triples = None
        self.graph = None
        self.norm = None
        self.label = None

    def sampling(self, pos_triples):
        """
        合并具有相同
        """
        batch_data = {}

        pos_triples = np.array(pos_triples)
        pos_triples, self.entity = self.sampling_positive(pos_triples)
        head_triples = self.sampling_negative('head', pos_triples, self.args.num_neg)
        tail_triples = self.sampling_negative('tail', pos_triples, self.args.num_neg)
        self.triples = np.concatenate((pos_triples, head_triples, tail_triples))
        batch_data['entity'] = self.entity
        batch_data['triples'] = self.triples

        self.label = torch.zeros((len(self.triples), 1))
        self.label[0: self.args.train_bs] = 1
        batch_data['label'] = self.label

        split_size = int(self.args.train_bs * 0.5)
        graph_split_ids = np.random.choice(
            self.args.train_bs,
            size=split_size,
            replace=False
        )
        head, rela, tail = pos_triples.transpose()
        head = torch.tensor(head[graph_split_ids], dtype=torch.long).contiguous()
        rela = torch.tensor(rela[graph_split_ids], dtype=torch.long).contiguous()
        tail = torch.tensor(tail[graph_split_ids], dtype=torch.long).contiguous()
        self.graph, self.relation, self.norm = self.build_graph(len(self.entity), (head, rela, tail), -1)
        batch_data['graph'] = self.graph
        batch_data['relation'] = self.relation
        batch_data['norm'] = self.norm

        return batch_data

    def get_sampling_keys(self):
        return ['graph', 'triples', 'label', 'entity', 'relation', 'norm']

    def sampling_negative(self, mode, pos_triples, num_neg):
        """
        不带过滤的随机负采样
        参数：
            mode：负采样模式。
            pos_triples：正样本三元组。
            num_neg：每个三元组对应的负样本数量。
        结果：
            neg_samples：负样本三元组。
        """
        neg_random = np.random.choice(
            len(self.entity),
            size=num_neg * len(pos_triples)
        )
        neg_samples = np.tile(pos_triples, (num_neg, 1))
        if mode == 'head':
            neg_samples[:, 0] = neg_random
        elif mode == 'tail':
            neg_samples[:, 2] = neg_random
        return neg_samples

    def build_graph(self, num_ent, triples, power):
        """
        使用 DGL 中的 dgl.graph 函数，使用采样三元组构建图。
        参数：
            num_ent：实体数量。
            triples：正样本三元组。
            power：用于正则化的幂指数。
        返回：
            rela：采样三元组之间的关系。
            graph：DGL 中 dgl.graph 函数构建的采样三元组图。
            edge_norm：图中的边范数。
        """
        head, rela, tail = triples[0], triples[1], triples[2]
        graph = dgl.graph(([], []))
        graph.add_nodes(num_ent)
        graph.add_edges(head, tail)
        node_norm = self.comp_deg_norm(graph, power)
        edge_norm = self.node_norm_to_edge_norm(graph, node_norm)
        rela = torch.tensor(rela)
        return graph, rela, edge_norm

    def comp_deg_norm(self, graph, power=-1):
        """
        计算归一化节点权重。
        参数：
            graph：DGL 中由 dgl.graph 构建的采样三元组图。
            power：归一化的幂指数。
        返回：
            tensor：归一化的节点权重。
        """
        graph = graph.local_var()
        in_deg = graph.in_degrees(range(graph.number_of_nodes())).float().numpy()
        norm = in_deg.__pow__(power)
        norm[np.isinf(norm)] = 0
        return torch.from_numpy(norm)

    def node_norm_to_edge_norm(slef, graph, node_norm):
        """
        计算归一化边权重。
        参数：
            graph：DGL 中由 dgl.graph 采样的三元组构成的图。
            node_norm：归一化的节点权重。
        返回：
            tensor：归一化的边权重。
        """
        graph = graph.local_var()
        # convert to edge norm
        graph.ndata['norm'] = node_norm.view(-1, 1)
        graph.apply_edges(lambda edges: {'norm': edges.dst['norm']})
        return graph.edata['norm']

    def sampling_positive(self, positive_triples):
        """
        重新生成正样本。
        参数：
            positive_triples：正样本三元组。
        结果：
            重新生成的三元组和实体会过滤掉不可见的实体。
        """

        edges = np.random.choice(
            np.arange(len(positive_triples)),
            size=self.args.train_bs,
            replace=False
        )
        edges = positive_triples[edges]
        head, rela, tail = np.array(edges).transpose()
        entity, index = np.unique((head, tail), return_inverse=True)
        head, tail = np.reshape(index, (2, -1))

        return np.stack((head, rela, tail)).transpose(), \
               torch.from_numpy(entity).view(-1, 1).long()


class KBATSampler(BaseSampler):
    """
    基于图的神经网络中的 n_hop 邻居。
    属性：
        n_hop：n_hop 邻居的图。
        graph：邻接图。
        neighbors：采样三元组的邻居。
        adj_matrix：采样的三元组。
        triples：采样的三元组。
        triples_GAT_pos：正三元组。
        triples_GAT_neg：负三元组。
        triples_Con：所有三元组，包括正三元组和负三元组。
        label：将假尾部掩蔽为负样本。
    """

    def __init__(self, args):
        super().__init__(args)
        self.n_hop = None
        self.graph = None
        self.neighbours = None
        self.adj_matrix = None
        self.entity = None
        self.triples_GAT_pos = None
        self.triples_GAT_neg = None
        self.triples_Con = None
        self.label = None

        self.get_neighbors()

    def sampling(self, pos_triples):
        """
        神经网络中基于图的 n_hop 邻居。
        参数：
            pos_triples：用于采样的三元组。
        返回：
            batch_data：训练数据。
        """
        batch_data = {}
        # --------------------KBAT-Sampler------------------------------------------
        self.entity = self.get_unique_entity(pos_triples)
        head_triples = self.sam_negative('head', pos_triples, self.args.num_neg)
        tail_triples = self.sam_negative('tail', pos_triples, self.args.num_neg)
        self.triples_GAT_neg = torch.tensor(np.concatenate((head_triples, tail_triples)))
        batch_data['triples_GAT_pos'] = torch.tensor(pos_triples)
        batch_data['triples_GAT_neg'] = self.triples_GAT_neg

        head, rela, tail = torch.tensor(self.train_triples).t()
        self.adj_matrix = (torch.stack((tail, head)), rela)
        batch_data['adj_matrix'] = self.adj_matrix

        self.n_hop = self.get_batch_nhop_neighbors_all()
        batch_data['n_hop'] = self.n_hop
        # --------------------ConvKB-Sampler------------------------------------------
        head_triples = self.sampling_negative('head', pos_triples, self.args.num_neg)
        tail_triples = self.sampling_negative('tail', pos_triples, self.args.num_neg)
        self.triples_Con = np.concatenate((pos_triples, head_triples, tail_triples))
        self.label = -torch.ones((len(self.triples_Con), 1))
        self.label[0: self.args.train_bs] = 1
        batch_data['triples_Con'] = self.triples_Con
        batch_data['label'] = self.label

        return batch_data

    def get_sampling_keys(self):
        return ['adj_matrix', 'n_hop', 'triples_GAT_pos',
                'triples_GAT_neg', 'triples_Con', 'label']

    def bfs(self, graph, source, nbd_size=2):
        """
        使用深度优先搜索算法生成 n_hop 邻接图。
        参数：
            graph：邻接图。
            source：头节点。
            nbd_size：跳数。
        返回：
            neighbors：N_hop 邻接图。
        """
        visit = {}
        distance = {}
        parent = {}
        distance_lengths = {}

        visit[source] = 1
        distance[source] = 0
        parent[source] = (-1, -1)

        q = queue.Queue()
        q.put((source, -1))

        while (not q.empty()):
            top = q.get()
            if top[0] in graph.keys():
                for target in graph[top[0]].keys():
                    if (target in visit.keys()):
                        continue
                    else:
                        q.put((target, graph[top[0]][target]))

                        distance[target] = distance[top[0]] + 1

                        visit[target] = 1
                        if distance[target] > 2:
                            continue
                        parent[target] = (top[0], graph[top[0]][target])  # 记录父亲节点id和关系id

                        if distance[target] not in distance_lengths.keys():
                            distance_lengths[distance[target]] = 1

        neighbors = {}
        for target in visit.keys():
            if (distance[target] != nbd_size):
                continue
            edges = [-1, parent[target][1]]
            relations = []
            entities = [target]
            temp = target
            while (parent[temp] != (-1, -1)):
                relations.append(parent[temp][1])
                entities.append(parent[temp][0])
                temp = parent[temp][0]

            if (distance[target] in neighbors.keys()):
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1])))  # 删除已知的source 记录前两跳实体及关系
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]

        return neighbors

    def get_neighbors(self, nbd_size=2):
        """
        获取源在 n_hop 邻域内的关系和实体。
        参数：
            nbd_size：跳数。
        返回：
            self.neighbours：记录源在 n_hop 邻域内的关系和实体。
        """
        self.graph = {}

        for triple in self.train_triples:
            head = triple[0]
            rela = triple[1]
            tail = triple[2]

            if (head not in self.graph.keys()):
                self.graph[head] = {}
                self.graph[head][tail] = rela
            else:
                self.graph[head][tail] = rela

        neighbors = {}
        '''
        import pickle
        print("Opening node_neighbors pickle object")
        file = self.args.data_path + "/2hop.pickle"
        with open(file, 'rb') as handle:
            self.neighbours = pickle.load(handle)  
        return
        '''
        start_time = time.time()
        print("Start Graph BFS")
        for head in self.graph.keys():
            temp_neighbors = self.bfs(self.graph, head, nbd_size)
            for distance in temp_neighbors.keys():
                if (head in neighbors.keys()):
                    if (distance in neighbors[head].keys()):
                        neighbors[head][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors[head][distance] = temp_neighbors[distance]
                else:
                    neighbors[head] = {}
                    neighbors[head][distance] = temp_neighbors[distance]

        print("Finish BFS, time taken ", time.time() - start_time)
        self.neighbours = neighbors

    def get_unique_entity(self, triples):
        """
        获取实体集合。

        参数：
            triples：采样的三元组。

        返回：
            numpy.array：实体集合
        """
        train_triples = np.array(triples)
        train_entities = np.concatenate((train_triples[:, 0], train_triples[:, 2]))
        return np.unique(train_entities)

    def get_batch_nhop_neighbors_all(self, nbd_size=2):
        """
        获取批量中所有实体的 n_hop 个邻居。

        参数：
            nbd_size：跳数。

        返回：
            n_hop 个邻居的集合。
        """
        batch_source_triples = []

        for source in self.entity:
            if source in self.neighbours.keys():
                nhop_list = self.neighbours[source][nbd_size]
                for i, tup in enumerate(nhop_list):
                    if (self.args.partial_2hop and i >= 2):
                        break
                    batch_source_triples.append([source,
                                                 tup[0][-1],
                                                 tup[0][0],
                                                 tup[1][0]])

        n_hop = np.array(batch_source_triples).astype(np.int32)

        return torch.autograd.Variable(torch.LongTensor(n_hop))

    def sampling_negative(self, mode, pos_triples, num_neg):
        """
        随机负采样。
        参数：
            mode：负采样模式。
            pos_triples：正样本三元组。
            num_neg：每个三元组对应的负样本数量。
        结果：
            neg_samples：负样本三元组。
        """
        neg_samples = np.tile(pos_triples, (num_neg, 1))
        if mode == 'head':
            neg_head = []
            for h, r, t in pos_triples:
                neg_head.append(self.head_batch(h, r, t, num_neg))
            neg_samples[:, 0] = torch.tensor(neg_head).t().reshape(-1)
        elif mode == 'tail':
            neg_tail = []
            for h, r, t in pos_triples:
                neg_tail.append(self.tail_batch(h, r, t, num_neg))
            neg_samples[:, 2] = torch.tensor(neg_tail).t().reshape(-1)
        return neg_samples

    def sam_negative(self, mode, pos_triples, num_neg):
        """
        无过滤器的随机负采样。
        参数：
            mode：负采样模式。
            pos_triples：正样本三元组。
            num_neg：每个三元组对应的负样本数量。
        结果：
            neg_samples：负样本三元组。
        """
        neg_random = np.random.choice(
            len(self.entity),
            size=num_neg * len(pos_triples)
        )
        neg_samples = np.tile(pos_triples, (num_neg, 1))
        if mode == 'head':
            neg_samples[:, 0] = neg_random
        elif mode == 'tail':
            neg_samples[:, 2] = neg_random
        return neg_samples


class CompGCNSampler(GraphSampler):
    """
    神经网络中基于图的采样。
    属性：
        relation：采样三元组之间的关系。
        triples：采样三元组。
        graph：DGL 中由 dgl.graph 构建的采样三元组图。
        norm：图中的边范数。
        label：将假尾部掩码为负样本。
    """

    def __init__(self, args):
        super().__init__(args)
        self.relation = None
        self.triples = None
        self.graph = None
        self.norm = None
        self.label = None

        super().get_hr_trian()

        self.graph, self.relation, self.norm = \
            self.build_graph(self.args.num_ent, np.array(self.t_triples).transpose(), -0.5)

    def sampling(self, pos_hr_t):
        """
        神经网络中基于图的 n_hop 邻居。
        参数：
            pos_hr_t：用于采样的三元组 (hr, t)。
        返回：
            batch_data：训练数据。
        """
        batch_data = {}

        self.label = torch.zeros(self.args.train_bs, self.args.num_ent)
        self.triples = torch.LongTensor([hr for hr, _ in pos_hr_t])
        for id, hr_sample in enumerate([t for _, t in pos_hr_t]):
            self.label[id][hr_sample] = 1

        batch_data['sample'] = self.triples
        batch_data['label'] = self.label
        batch_data['graph'] = self.graph
        batch_data['relation'] = self.relation
        batch_data['norm'] = self.norm

        return batch_data

    def get_sampling_keys(self):
        return ['sample', 'label', 'graph', 'relation', 'norm']

    def node_norm_to_edge_norm(self, graph, node_norm):
        """
        计算归一化边权重。
        参数：
            graph：DGL 中由 dgl.graph 抽样的三元组构成的图。
            node_norm：归一化的节点权重。
        返回：
            norm：归一化的边权重。
        """
        graph.ndata['norm'] = node_norm
        graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        norm = graph.edata.pop('norm').squeeze()
        return norm


class TestSampler(object):
    """
    对三元组进行采样并记录用于测试的正三元组。
        属性：
        sampler：训练采样器的函数。
        hr2t_all：记录对应于相同头和关系的尾部。
        rt2h_all：记录对应于相同尾部和关系的头。
        ​​num_ent：实体数量。
    """

    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.num_ent = sampler.args.num_ent

    def get_hr2t_rt2h_from_all(self):
        """
        从所有数据集（训练集、验证集和测试集）中获取 hr2t 和 rt2h 的集合，数据类型为张量。
        更新：
            self.hr2t_all：hr2t 的集合。
            self.rt2h_all：rt2h 的集合。
        """
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        """
        采样三元组并记录用于测试的正三元组。
        参数：
            data：用于采样的三元组。
        返回：
            batch_data：用于评估的数据。
        """
        batch_data = {}
        head_label = torch.zeros(len(data), self.num_ent)
        tail_label = torch.zeros(len(data), self.num_ent)
        for idx, triple in enumerate(data):
            head, rel, tail = triple
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "head_label", "tail_label"]


class GraphTestSampler(object):
    """
    用于测试的采样图。
    属性：
        sampler：训练采样器的功能。
        hr2t_all：记录相同头部和关系对应的尾部。
        rt2h_all：记录相同尾部和关系对应的头部。
        num_ent：实体数量。
        triples：训练三元组。
    """

    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.num_ent = sampler.args.num_ent
        self.triples = sampler.train_triples

    def get_hr2t_rt2h_from_all(self):
        """
        从所有数据集（训练集、验证集和测试集）中获取 hr2t 和 rt2h 的集合，数据类型为张量。
        更新：
            self.hr2t_all：hr2t 的集合。
            self.rt2h_all：rt2h 的集合。
        """
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        """
        用于测试的采样图。
        参数：
            data：用于采样的三元组。
        返回：
            batch_data：用于评估的数据。
        """
        batch_data = {}
        head_label = torch.zeros(len(data), self.num_ent)
        tail_label = torch.zeros(len(data), self.num_ent)
        for idx, triple in enumerate(data):
            # from IPython import embed;embed();exit()
            head, rel, tail = triple
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label

        head, rela, tail = np.array(self.triples).transpose()
        graph, rela, norm = self.sampler.build_graph(self.num_ent, (head, rela, tail), -1)
        batch_data["graph"] = graph
        batch_data["rela"] = rela
        batch_data["norm"] = norm
        batch_data["entity"] = torch.arange(0, self.num_ent, dtype=torch.long).view(-1, 1)

        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "head_label", "tail_label", \
                "graph", "rela", "norm", "entity"]


class CompGCNTestSampler(object):
    """
    用于测试的采样图。
    属性：
        sampler：训练采样器的功能。
        hr2t_all：记录相同头部和关系对应的尾部。
        rt2h_all：记录相同尾部和关系对应的头部。
        num_ent：实体数量。
        triples：训练三元组。
    """

    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.num_ent = sampler.args.num_ent
        self.triples = sampler.t_triples

    def get_hr2t_rt2h_from_all(self):
        """
        从所有数据集（训练集、验证集和测试集）中获取 hr2t 和 rt2h 的集合，数据类型为张量。

        更新：
            self.hr2t_all：hr2t 的集合。
            self.rt2h_all：rt2h 的集合。
        """
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        """
        用于测试的采样图。
        参数：
            data：用于采样的三元组。
        返回：
            batch_data：用于评估的数据。
        """
        batch_data = {}

        head_label = torch.zeros(len(data), self.num_ent)
        tail_label = torch.zeros(len(data), self.num_ent)

        for idx, triple in enumerate(data):
            # from IPython import embed;embed();exit()
            head, rel, tail = triple
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label

        graph, relation, norm = \
            self.sampler.build_graph(self.num_ent, np.array(self.triples).transpose(), -0.5)

        batch_data["graph"] = graph
        batch_data["rela"] = relation
        batch_data["norm"] = norm
        batch_data["entity"] = torch.arange(0, self.num_ent, dtype=torch.long).view(-1, 1)

        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "head_label", "tail_label", \
                "graph", "rela", "norm", "entity"]


class SEGNNTrainProcess(RevSampler):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.use_weight = self.args.use_weight
        # Parameters when constructing graph
        self.src_list = []
        self.dst_list = []
        self.rel_list = []
        self.hr2eid = ddict(list)
        self.rt2eid = ddict(list)

        self.ent_head = []
        self.ent_tail = []
        self.rel = []

        self.query = []
        self.label = []
        self.rm_edges = []
        self.set_scaling_weight = []

        self.hr2t_train_1 = ddict(set)
        self.ht2r_train_1 = ddict(set)
        self.rt2h_train_1 = ddict(set)
        self.get_h2rt_t2hr_from_train()
        self.construct_kg()
        self.get_sampling()

    def get_h2rt_t2hr_from_train(self):
        for h, r, t in self.train_triples:
            if r <= self.args.num_rel:
                self.ent_head.append(h)
                self.rel.append(r)
                self.ent_tail.append(t)
                self.hr2t_train_1[(h, r)].add(t)
                self.rt2h_train_1[(r, t)].add(h)

        for h, r in self.hr2t_train:
            self.hr2t_train_1[(h, r)] = np.array(list(self.hr2t_train[(h, r)]))
        for r, t in self.rt2h_train:
            self.rt2h_train_1[(r, t)] = np.array(list(self.rt2h_train[(r, t)]))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        h, r, t = self.query[item]
        label = self.get_onehot_label(self.label[item])

        rm_edges = torch.tensor(self.rm_edges[item], dtype=torch.int64)
        rm_num = math.ceil(rm_edges.shape[0] * self.args.rm_rate)
        rm_inds = torch.randperm(rm_edges.shape[0])[:rm_num]
        rm_edges = rm_edges[rm_inds]

        return (h, r, t), label, rm_edges

    def get_onehot_label(self, label):
        onehot_label = torch.zeros(self.args.num_ent)
        onehot_label[label] = 1
        if self.args.label_smooth != 0.0:
            onehot_label = (1.0 - self.args.label_smooth) * onehot_label + (1.0 / self.args.num_ent)

        return onehot_label

    def get_sampling(self):
        for k, v in self.hr2t_train_1.items():
            self.query.append((k[0], k[1], -1))
            self.label.append(list(v))
            self.rm_edges.append(self.hr2eid[k])

        for k, v in self.rt2h_train_1.items():
            self.query.append((k[1], k[0] + self.args.num_rel, -1))
            self.label.append(list(v))
            self.rm_edges.append(self.rt2eid[k])

    def construct_kg(self, directed=False):
        """
        构建 kg。
        :param targeted: 是否为每条边添加逆版本，以构建无向图。
        训练 SE-GNN 模型时为 False，计算 SE 指标时为 True。
        :return:
        """

        # eid: record the edge id of queries, for randomly removing some edges when training
        eid = 0
        for h, t, r in zip(self.ent_head, self.ent_tail, self.rel):
            if directed:
                self.src_list.extend([h])
                self.dst_list.extend([t])
                self.rel_list.extend([r])
                self.hr2eid[(h, r)].extend([eid])
                self.rt2eid[(r, t)].extend([eid])
                eid += 1
            else:
                # include the inverse edges
                # inverse rel id: original id + rel num
                self.src_list.extend([h, t])
                self.dst_list.extend([t, h])
                self.rel_list.extend([r, r + self.args.num_rel])
                self.hr2eid[(h, r)].extend([eid, eid + 1])
                self.rt2eid[(r, t)].extend([eid, eid + 1])
                eid += 2

        self.src_list, self.dst_list, self.rel_list = torch.tensor(self.src_list), torch.tensor(
            self.dst_list), torch.tensor(self.rel_list)


class SEGNNTrainSampler(object):
    def __init__(self, args):
        self.args = args
        self.get_train_1 = SEGNNTrainProcess(args)
        self.get_valid_1 = SEGNNTrainProcess(args).get_valid()
        self.get_test_1 = SEGNNTrainProcess(args).get_test()

    def get_train(self):
        return self.get_train_1

    def get_valid(self):
        return self.get_valid_1

    def get_test(self):
        return self.get_test_1

    def sampling(self, data):
        src = [d[0][0] for d in data]
        rel = [d[0][1] for d in data]
        dst = [d[0][2] for d in data]
        label = [d[1] for d in data]  # list of list
        rm_edges = [d[2] for d in data]

        src = torch.tensor(src, dtype=torch.int64)
        rel = torch.tensor(rel, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)
        label = torch.stack(label, dim=0)
        rm_edges = torch.cat(rm_edges, dim=0)

        return (src, rel, dst), label, rm_edges


class SEGNNTestSampler(Dataset):
    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler
        # Parameters when constructing graph
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()

    def get_hr2t_rt2h_from_all(self):
        """
        从所有数据集（训练集、验证集和测试集）中获取 hr2t 和 rt2h 的集合，数据类型为张量。
        更新：
            self.hr2t_all：hr2t 的集合。
            self.rt2h_all：rt2h 的集合。
        """
        for h, r, t in self.sampler.get_train_1.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            # self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        # for r, t in self.rt2h_all:
        #     self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        """
        采样三元组并记录用于测试的正三元组。
        参数：
            data：用于采样的三元组。
        返回：
            batch_data：用于评估的数据。
        """
        batch_data = {}
        head_label = torch.zeros(len(data), self.sampler.args.num_ent)
        tail_label = torch.zeros(len(data), self.sampler.args.num_ent)
        filter_head = torch.zeros(len(data), self.sampler.args.num_ent)
        filter_tail = torch.zeros(len(data), self.sampler.args.num_ent)
        for idx, triple in enumerate(data):
            head, rel, tail = triple
            filter_tail[idx][self.hr2t_all[(head, rel)]] = -float('inf')
            filter_tail[idx][tail] = 0

            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)

        batch_data["filter_tail"] = filter_tail

        batch_data["tail_label"] = tail_label
        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "filter_tail", "tail_label"]


'''继承torch.Dataset'''


class KGDataset(Dataset):

    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]