# -*- codeing = utf-8 -*-
# @Time : 2025/7/8 16:53
# @Author : Luo_CW
# @File : run_DataPreprocess.py
# @Software : PyCharm
from DataPreprocess import KGData,BaseSampler
from setup_parser import SetupParser
from sampler import UniSampler


def run():
    parser = SetupParser()  # 设置参数
    args = parser.parse_args()
    kgdata = KGData(args)
    kgdata.get_hr2t_rt2h_from_train()
    # print(kgdata.train_triples)
    # print(len(kgdata.train_triples))
    unisampler = UniSampler(args)
    unisampler.uni_sampling(kgdata.train_triples)


if __name__ == '__main__':
    run()
