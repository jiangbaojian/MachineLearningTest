# -*- coding:utf-8 -*-
#__author__: 'Baojian Jiang'
#__time__:'2019/3/3'

# C和 sigma

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
import matplotlib.colors
from sklearn.metrics import accuracy_score


np.random.seed(0)
x = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = [0] * 20 + [1] * 20
print('------------------------------')
clf_param = (('linear', 0.1), ('linear', 0.5), ('linear', 1), ('linear', 2),
                ('rbf', 1, 0.1), ('rbf', 1, 1), ('rbf', 1, 10), ('rbf', 1, 100),
                ('rbf', 5, 0.1), ('rbf', 5, 1), ('rbf', 5, 10), ('rbf', 5, 100))
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
x1_min, x2_min = np.min(x, axis=0)
x1_max, x2_max = np.max(x, axis=0)
x1, x2 = np.mgrid[x1_min: x1_max: 200j, x2_min:x2_max:200j]
print('xxxxxxxxxxxxxx')
print(x1)
print('xxxxxxxxxxxxxxx')
grid_test = np.stack((x1.flat, x2.flat), axis=1)
cm_light = mpl.colors.ListedColormap(['#0894A1', '#47AB6C'])
cm_dark = mpl.colors.ListedColormap(['g', 'r'])
plt.figure(figsize=[13, 9], facecolor='w')
for i, param in enumerate(clf_param):
    clf = svm.SVC(C=param[1], kernel=param[0])
    if param[0] == 'rbf':
        clf.gamma = param[2]
        title = '高斯核，C=%.1f，$\gamma$ =%.1f' % (param[1], param[2])
    else:
        title = '线性核，C=%.1f' % param[1]
    clf.fit(x, y)
    y_hat = clf.predict(x)
    print('准确率：', accuracy_score(y, y_hat))
    print(title)
    print('支撑向量的数目：', clf.n_support_)
    print('支撑向量的系数：', clf.dual_coef_)
    print('支撑向量：', clf.support_)
    plt.subplot(3, 4, i+1)
    grid_hat = clf.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], grid_hat, cmap=cm_dark, edgecolors='k')
    plt.scatter(x[clf.support_, 0], x[clf.support_, 1], edgecolors='k', facecolors='none', s=100, marker='o')
    z = clf.decision_function(grid_test) #样本到每个类的距离
    print('clf.decision_function(x) = ', clf.decision_function(x))
    print('clf.predict(x) = ', clf.predict(x))
    z = z.reshape(x1.shape)
    print('xxxxxxxxxxxxxxxxxx')
    print(x1)
    print('xxxxxxxxxxxxxxx')
    plt.contour(x1, x2, z, colors=list('kbrbk'), linestyles=['--', '--', '-', '--', '--'],
                linewidths=[1, 0.5, 1.5, 0.5, 1], levels=[-1, -0.5, 0, 0.5, 1])
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(title, fontsize=12)
plt.suptitle('SVM不同参数的分类', fontsize=16)
plt.tight_layout(1.4)
plt.subplots_adjust(top=0.92)
plt.show()






