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




