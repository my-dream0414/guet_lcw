# -*- codeing = utf-8 -*-
# @Time : 2025/7/8 16:49
# @Author : Luo_CW
# @File : setup_parser.py
# @Software : PyCharm
import argparse
def SetupParser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data_path', default='../Dataset2/FB15K237', type=str, help='The name of dataset.')
    parser.add_argument('--use_weight', default=False, action='store_true', help='Use subsampling weight.')
    parser.add_argument('--filter_flag', default=True, action='store_false', help='Filter in negative sampling.')
    parser.add_argument('--num_neg', default=10, type=int,
                        help='The number of negative samples corresponding to each positive sample')
    parser.add_argument('--num_ent', default=None, type=int, help='The number of entity, autogenerate.')
    parser.add_argument('--num_rel', default=None, type=int, help='The number of relation, autogenerate.')
    parser.add_argument('--cuda', default=True, action='store_true', help='使用GPU')
    parser.add_argument('--max_steps', type=int, default=5000, help='最大训练步数')
    # # 网络结构参数
    parser.add_argument('--entity_dim', default=256, type=int, help='实体向量的维度.')
    parser.add_argument('--relation_dim', default=256, type=int, help='实体向量的维度.')
    parser.add_argument('--aggregation_type', default=None, type=int, help='实体向量的维度.')
    parser.add_argument('--mess_dropout', default=None, type=int, help='')
    parser.add_argument('--conv_dim_list', default=None, type=int, help='')
    parser.add_argument('--kg_l2loss_lambde', default=None, type=int, help='')
    # parser.add_argument('--', default=None, type=int, help='')
    # parser.add_argument('--', default=None, type=int, help='')
    # parser.add_argument('--', default=None, type=int, help='')
    return parser