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

    return parser