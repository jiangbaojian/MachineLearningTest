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
            # print(i[1])
            entropy += i[1] / dataset_len * np.log2(i[1]/dataset_len)
        return -entropy

    def createDataSet(self):
        # dataset = [[1, 1, 'yes'],
        #            [1, 1, 'yes'],
        #            [1, 0, 'no'],
        #            [0, 1, 'no'],
        #            [0, 1, 'no']]
        # labels = ['no surfacing', 'flippers']
        dataset = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
            ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
            ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
            ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
            ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
            ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
            ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
            ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
            ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
            ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
            ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
            ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
            ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
        ]

        # 特征值列表
        labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']
        return dataset, labels

    def chooseBestFeature(self, dataset):
        root_entropy = self.calcEntropy(dataset)
        characters = len(dataset[0]) - 1
        dataset_length = len(dataset)
        infor_gain = {}
        for character in range(characters):#遍历整个属性
            character_flag = {}
            for i in dataset:#找出每一个属性的不同取值的列表
                if i[character] not in character_flag.keys():
                    character_flag[i[character]] = 1
                else:
                    character_flag[i[character]] += 1
            branch_entropy = 0.0
            for key,value in character_flag.items():#分别求出每一个列表的熵
                sub_dataset = [i for i in dataset if i[character] == key] #find D1 dataset
                prob = len(sub_dataset) / float(dataset_length)
                branch_entropy += prob * self.calcEntropy(sub_dataset)
            infor_gain[character] = root_entropy - branch_entropy
        print("Entropy for each characters:", infor_gain)
        best_brahch = 0
        gain = 0.0
        for i in range(1, len(infor_gain)):
            if infor_gain[i] > infor_gain[i-1]:
                best_brahch = i
                gain = infor_gain[i]
        return best_brahch, gain


    def splitDataSet(self, dataset, axis, value):
        retDataSet = []
        for featVec in dataset:
            if featVec[axis] ==value:
                reduceFeatVec = featVec[:axis]
                reduceFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reduceFeatVec)
        return retDataSet
if __name__  == '__main__':
    dt = DecisionTree()
    dataset, labels = dt.createDataSet()
    dataset = np.array(dataset)
    entropy = dt.calcEntropy(dataset) #获取信息熵
    print('计算信息熵：', entropy)
    best_branch, gain = dt.chooseBestFeature(dataset)
    print(best_branch, gain)


