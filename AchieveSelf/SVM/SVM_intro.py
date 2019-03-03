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
    x = x[y != 0, :2] #做成二分类问题
    y = y[y != 0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
    for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
        clf = svm.SVC(C=0.1, kernel=kernel, gamma=10, decision_function_shape='ovr')
        clf.fit(x_train, y_train)
        plt.figure(fig_num)
        plt.clf()
        plt.scatter(x[:, 0], x[:, 1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolors='k', s=20)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=80, facecolors='none', zorder=10, edgecolors='k')
        plt.axis('tight')
        x0_min, x0_max = x[:, 0].min(), x[:, 0].max()
        x1_min, x1_max = x[:, 1].min(), x[:, 1].max()
        x0, x1 = np.mgrid[x0_min: x0_max: 200j, x1_min:x1_max:200j]
        z = clf.decision_function(np.c_[x0.ravel(), x1.ravel()])
        z = z.reshape(x0.shape)
        plt.pcolormesh(x0, x1, z > 0, cmap=plt.cm.Paired)
        plt.contour(x0, x1, z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
        plt.title(kernel)
    plt.show()





    '''
    x = pd.DataFrame(x)
    x = x[[0, 1]]
    print(x)
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
    # x1_min, x2_min = x[:, 0].min(), x[:, 1].min()
    # x1_max, x2_max = x[:, 0].max(), x[:, 1].max()

    x1_min, x2_min = x.min()
    x1_max, x2_max = x.max()
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 生成网格采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    grid_hat = clf.predict(grid_test)  # 预测分类值
    grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
    mpl.rcParams['font.sans-serif'] = ['SimHei']
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
    plt.title('鸢尾花SVM二特征分类', fontsize=16)
    plt.grid(b=True, ls=':')
    plt.tight_layout(pad=1.5)
    plt.show()'''



