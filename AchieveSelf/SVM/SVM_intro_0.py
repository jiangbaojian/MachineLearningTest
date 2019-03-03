# -*- coding:utf-8 -*-
#__author__: 'Baojian Jiang'
#__time__:'2019/3/3'

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


x, y = make_blobs(n_samples=40, centers=2, random_state=6)
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
clf = svm.SVC(C=1, kernel='linear', gamma=0.001)
clf.fit(x, y)
plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
x1 = np.linspace(xlim[0], xlim[1], 30)
y1 = np.linspace(ylim[0], ylim[1], 30)
y1, x1 = np.meshgrid(y1, x1)
x1y1 = np.vstack([x1.ravel(), y1.ravel()]).T
z = clf.decision_function(x1y1).reshape(x1.shape) #样本点到分隔超平面的函数距离
print(z)
ax.contour(x1, y1, z, color='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()