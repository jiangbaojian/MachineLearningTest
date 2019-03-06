# -*- coding:utf-8 -*-
#__author__: 'Baojian Jiang'
#__time__:'2019/3/4'


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# create two clusters of random points
n_samples_1 = 1000
n_samples_2 = 100
centers = [[0.0, 0.0], [2.0, 2.0]]
clusters_std = [1.5, 0.5]
x, y = make_blobs(n_samples=[n_samples_1, n_samples_2],
                  centers=centers,
                  cluster_std=clusters_std,
                  random_state=0, shuffle=False)

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(x, y)
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(x, y)
plt.scatter(x[:, 0]. x[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_yaxi()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
yy, xx = np.meshgrid(yy, xx)
xy = np.vstack([xx.ravel(), yy.ravel()])
z = clf.decision_function(xy).reshape(xx.shape)
a = ax.contour(xx, yy, z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
z = wclf.decision_function(xy).reshape(xx.shape)
b = ax.contour(xx, yy, z, colors='r', levels=[0], alpha=0.5, linestyles=['-'])
plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"],
           loc="upper right")
plt.show()



