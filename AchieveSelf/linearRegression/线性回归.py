# -*- coding: utf-8 -*-
# __author__: 'Baojian Jiang'
# __date__: 2018年10月22日14:58:42
#对数回归

from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sympy import *
import numpy as np
import time
import random
from sklearn.linear_model import Lasso

def loadDataSet(file_name):
    """load data
    :param fileName: data file path
    :return: dataMat,labelMat
    """
    fr = open(file_name)
    numFeat = len(fr.readline().split('\t')) - 1  #the number of data each line
    dataMat = []
    labelMat = []
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

if __name__ == '__main__':
    '''
    data_file_path = '30a.xlsx'
    data = pd.read_excel(data_file_path)
    # 获取列的数据
    x = data['密度']
    y = data['含糖度']
    target = ['好瓜']
    plt.scatter(x,y)
    plt.show()
    '''
    '''
    x, m, y = symbols('x,m,y')
    # y = 1 / (1 + x ** 2 + m ** 2)
    f = sin(x*y*m)
    print(diff(f, y))#求导,
    '''
    start_time = time.time()
    file_path = 'AchieveSelf/ex0.txt'
    x, y = loadDataSet(file_path)
    xMat = np.mat(x)
    # print('xxxxmat', xMat)
    # print(np.shape(xMat))
    # print('ones',np.ones((2, 1)))
    # x = np.array(x)
    # sample_num, dim = x.shape
    # weights = np.mat(np.eye(sample_num))
    # print(weights)
    # print(type(weights))
    # weight = np.eye(sample_num)
    # print(weight)
    # print(type(weight))
    #虽然看着相同，但是一个是矩阵一个是数组，类型不同
    # print('样本个数', sample_num)
    # print('样本属性', dim)
    # print(np.zeros((dim,)))
    # w = np.ones((dim, ), dtype=np.float32)
    # print(w)
    # print(x.shape)#返回列表的形状
    # print(x.flatten())#将不论几维的列表变为一维
    # print(np.zeros(2))
    # index = random.sample(range(sample_num), int(np.ceil(sample_num * 0.7)))
    # 从这些里面随机取140个
    # print(np.ceil(sample_num * 0.7))
    # print(len(index))
    # print(index)
    n = len(x[0])
    weights = []
    for i in range(n):
        weights.append(1)
    weights = np.array(weights)
    print('weights', weights)
    print('x0', x[0])
    print((weights * x[0]).sum())
    print('n,1', np.ones((3, 1)))
    end_time = time.time()
    print('程序运行时间', end_time-start_time)