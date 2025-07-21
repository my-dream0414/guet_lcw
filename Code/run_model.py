# -*- codeing = utf-8 -*-
# @Time : 2025/7/10 20:00
# @Author : Luo_CW
# @File : run_model.py
# @Software : PyCharm
from DataPreprocess import KGData,BaseSampler
from setup_parser import SetupParser
from sampler import UniSampler
from KGAT import KGAT


def run():
    parser = SetupParser()  # 设置参数
    args = parser.parse_args()
    kgdata = KGData(args)
    # kgdata处理后得到：train_triples、vaild_triples、test_triples

    unisampler = UniSampler(args)
    unisampler.uni_sampling(kgdata.train_triples)



if __name__ == '__main__':
    run()
