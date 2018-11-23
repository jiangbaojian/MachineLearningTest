# -*- coding:utf-8 -*-
#__author__: 'Baojian Jiang'
#__time__:'2018/11/23'
import numpy as np
class DecisionTree:
    """
    决策树的练习
    """
    def calcEntropy(self, dataset):
        """
        计算信息熵
        :param dataset:
        :return:
        """
        dataset = np.array(dataset)
        dataset_len = len(dataset)
        dataset_flag = {}
        for i in dataset:
            if i[-1] not in dataset_flag.keys():
                dataset_flag[i[-1]] = 1
            else:
                dataset_flag[i[-1]] += 1
        entropy = 0.0
        for i in dataset_flag.items():
            print(i[1])
            entropy += i[1] / dataset_len * np.log2(i[1]/dataset_len)
        return -entropy

    def createDataSet(self):
        dataset = [[1, 1, 'yes'],
                   [1, 1, 'yes'],
                   [1, 0, 'no'],
                   [0, 1, 'no'],
                   [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        return dataset, labels
if __name__  == '__main__':
    dt = DecisionTree()
    dataset, labels = dt.createDataSet()
    dt = dt.calcEntropy(dataset) #获取信息熵
    print('计算信息熵：', dt)

