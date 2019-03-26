# -*- coding:utf-8 -*-
#__author__: 'Baojian Jiang'
#__time__:'2019/2/16'

import numpy as np
import matplotlib as mpl
import  matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets.samples_generator import make_blobs

def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d



if __name__ == '__main__':
    N = 400
    centers = 4
    data1, y1 = make_blobs(N, n_features=2, centers=centers, random_state=2)
    data2, y2 = make_blobs(N, n_features=2, centers=centers, cluster_std=(1, 2.5, 0.5, 2), random_state=2)
    data3 = np.vstack((data1[y1 == 0][:100], data1[y1 == 1][:50], data1[y1 == 2][:20], data1[y1 == 3][:5]))
    y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)
    cls = KMeans(n_clusters=4, init='k-means++')
    y1_hat = cls.fit_predict(data1)
    y2_hat = cls.fit_predict(data2)
    y3_hat = cls.fit_predict(data3)

    m = np.array(((1, 1), (1, 3)))
    data1_r = data1.dot(m)
    y_r_hat = cls.fit_predict(data1_r)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm = mpl.colors.ListedColormap(list('rgbm'))
    plt.figure(figsize=(9, 10), facecolor='w')
    plt.subplot(421)
    plt.title('原始数据')
    plt.scatter(data1[:, 0], data1[:, 1], c=y1, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data1, axis=0)
    x1_max, x2_max = np.max(data1, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(422)
    plt.title('KMeans++聚类')
    plt.scatter(data1[:, 0], data1[:, 1], c=y1_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(423)
    plt.title('旋转后数据')
    plt.scatter(data1_r[:, 0], data1_r[:, 1], c=y1, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(424)
    plt.title('旋转后KMean++聚类')
    plt.scatter(data1_r[:, 0], data1_r[:, 1], c=y_r_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(425)
    plt.title('方差不等数据')
    plt.scatter(data2[:, 0], data2[:, 1], c=y2, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data2, axis=0)
    x1_max, x2_max = np.max(data2, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(426)
    plt.title('方差不等KMean++')
    plt.scatter(data2[:, 0], data2[:, 1], c=y2_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(427)
    plt.title('数量不相等数据')
    plt.scatter(data3[:, 0], data3[:, 1], s=30, c=y3, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data3, axis=0)
    x1_max, x2_max = np.max(data3, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(428)
    plt.title(u'数量不相等KMeans++聚类')
    plt.scatter(data3[:, 0], data3[:, 1], c=y3_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.tight_layout(2, rect=(0, 0, 1, 0.97))
    plt.suptitle(u'数据分布对KMeans聚类的影响', fontsize=18)

    plt.show()
