# -*- coding:utf-8 -*-
#__author__: 'Baojian Jiang'
#__time__:'2019/3/1'

import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris_feature = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']

if __name__ == '__main__':
    iris = load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
    clf = svm.SVC(C=0.1, kernel='rbf', decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())
    # 准确率
    print(clf.score(x_train, y_train)) #精度
    print('训练集准确率: ', accuracy_score(y_train, clf.predict(x_train)))
    print(clf.score(x_test, y_test))
    print('测试集准确度: ', accuracy_score(y_test, clf.predict(x_test)))
    # decision_function
    print('decision_function:', clf.decision_function(x_train))
    print('\n predict:\n', clf.predict(x_train))
    # 画图
    # x1_min, x2_min = x.min()
    # x1_max, x2_max = x.max()
    x1_min, x2_min = x[:, 0].min(), x[:, 1].min()
    x1_max, x2_max = x[:, 0].max(), x[:, 1].max()

    x1, x2 = np.mgrid[x1_min: x1_max:150j, x2_min: x2_max: 150j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    grid_hat = clf.predict(x)
    print(grid_hat)
    grid_hat = grid_hat.reshape(x1.shape)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)  # 样本
    plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
    plt.xlabel(iris_feature[0], fontsize=13)
    plt.ylabel(iris_feature[1], fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'鸢尾花SVM二特征分类', fontsize=16)
    plt.grid(b=True, ls=':')
    plt.tight_layout(pad=1.5)
    plt.show()



