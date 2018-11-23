# -*- coding:utf-8 -*-
#__author__: 'Baojian Jiang'
#__time__:'2018/11/7'
# 线性回归模型

import numpy as np
import matplotlib.pyplot as plt
import time
import random

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

def showData(x, y):
    x_data = []
    for i in x:
        x_data.append(i[1])
    plt.scatter(x_data, y, c='g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('data distribution')
    plt.ylim(3, 4.7)
    plt.show()

def showLogisticData(x, y):
    y0_index = []
    x = np.array(x)
    y = np.array(y)
    # for i in range(0, len(y)):
    #     if y[i] == 0.0:
    #         y0_index.append(i)
    y0_index = np.where(y == 0.0)
    y1_index = np.where(y == 1.0)
    print(x[y0_index])
    x0 = x[y0_index]
    x1 = x[y1_index]
    x0_0 = x0[:, 0]
    x0_1 = x0[:, 1]
    x1_0 = x1[:, 0]
    x1_1 = x1[:, 1]
    plt.plot(x0_0, x0_1, 'g^', x1_0, x1_1, 'bs')
    plt.xlabel('x0')
    plt.ylabel('y0')
    plt.show()

def LinearRegres(xArr, yArr):
    """
    for now just sample linear regression,no matrix reversible solution
    :param xArr:
    :param yArr:
    :return:
    """
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0:  #计算行列式
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def Batch_Gradient_Descent(dataMatIn, y, step_size=0.01, max_iter_count=10000):
    sample_num, dim = dataMatIn.shape  #sample_num,x的行数，dim x的列的数量
    y = y.flatten()
    theta = np.ones((dim,), dtype=np.float32)#这个相当于是theta0
    loss = 10
    iter_count = 0
    while loss > 0.001 and iter_count < max_iter_count:
        loss = 0
        error = np.zeros((dim,), dtype=np.float32)
        for i in range(sample_num):
            predict_y = np.dot(theta.T, dataMatIn[i]) #矩阵乘法,是否是第一次运行的结果
            for j in range(dim):
                error[j] += (y[i] - predict_y) * dataMatIn[i][j]
        for j in range(dim):
            theta[j] += step_size * error[j] / sample_num
        for i in range(sample_num):
            predict_y = np.dot(theta.T, dataMatIn[i])
            error = (1 / (sample_num * dim)) * np.power((predict_y - y[i]), 2)
            loss += error
        print('iter_count:', iter_count, 'the loss:', loss)
        iter_count += 1
    return theta

def bgd(x, y, alpha=0.001, max_iter_count=1000):
    '''批梯度下降算法
    批量处理，每一次更新都需要遍历所有的数据
    '''
    x_count, dim = x.shape
    y = y.flatten()
    theta = np.ones((dim,), dtype=np.float32) #theta初始值
    iter_count = 0
    loss_former = 10
    while loss_former > 0.01 and iter_count < max_iter_count:
        w = np.zeros((dim,), dtype=np.float32)
        loss = np.float32(0.0)
        for i in range(x_count):
            y_hat = np.dot(theta.T, x[i])
            for j in range(dim):
                w[j] += (y[i] - y_hat) * x[i][j] #因为可能不是一维的，所以出现的数据可能有多位
        for j in range(dim):
            # theta[j] += alpha * w[j] / x_count #不明白为什么需要除以样本数量,是方法不一样，这个用的是最小二乘法，其他的用的是MSE（均方误差
            theta[j] += alpha * w[j]
        for i in range(x_count):
            loss += np.power(y[i] - np.dot(theta.T, x[i]), 2) * 1 / 2
        loss = np.abs(loss - loss_former)
        loss_former =loss
        print('第 %s 次循环，theta值为: %s, loss值为: %f' % (iter_count, theta, loss))
        iter_count += 1
    return theta

def sgb(x, y, alpha=0.001, max_iter_count=10000):
    '''
    随机梯度下降，每次更新不必鞥更新所有数据，这值和初始theta值有关
    :param x:
    :param y:
    :param alpha:
    :param max_iter_count:
    :return:
    '''
    x_count, dim = x.shape
    y = y.flatten()
    theta = np.ones((dim,), dtype=np.float32)
    iter_count = 0
    loss_former = 10
    while loss_former > 0.01 and iter_count < max_iter_count:
        w = np.zeros((dim, ), dtype=np.float32)
        loss = np.float32(0.0)
        for i in range(x_count):
            y_hat = np.dot(theta.T, x[i])
            for j in range(dim):
                w[j] += (y[i] - y_hat) * x[i][j]
                theta[j] += alpha * w[j]
        for i in range(x_count):
            loss += np.power(y[i] - np.dot(theta.T, x[i]), 2) * 1 / 2
        loss = np.abs(loss - loss_former)
        loss_former = loss
        print('第 %s 次循环，theta值为: %s, loss值为: %f' % (iter_count, theta, loss))
        iter_count += 1
    return theta

def mbgb(x, y, alpha=0.01, max_iter_count=10000, bitch_size=0.2):
    '''
    随机梯度下降
    :param x:
    :param y:
    :param alpha:
    :param max_iter_count:
    :param bitch_size:
    :return:
    '''
    x_count, dim = x.shape
    y = y.flatten()
    theta = np.ones((dim,), dtype=np.float32)
    iter_count = 0
    loss_former = 10
    while iter_count < max_iter_count:
        w = np.zeros((dim, ), dtype=np.float32)
        loss = np.float32(0.0)
        index = random.sample(range(x_count), int(np.ceil(x_count * bitch_size)))
        batch_x = x[index]
        batch_y = y[index]
        for i in range(len(batch_x)):
            y_hat = np.dot(theta.T, batch_x[i])
            for j in range(dim):
                w[j] += (batch_y[i] - y_hat) * batch_x[i][j]
                theta[j] += alpha * w[j]
        for i in range(x_count):
            loss += np.power(y[i] - np.dot(theta.T, x[i]), 2) * 1 / 2
        loss = np.abs(loss - loss_former)
        loss_former = loss
        print('第 %s 次循环，theta值为: %s, loss值为: %f' % (iter_count, theta, loss))
        iter_count += 1
    return theta

def lwlr(testPoint, x, y, k=1.0):
    '''
    局部加权线性回归
    :param testPoint:样本点
    :param x: 输入x
    :param y: y
    :param k: 控制衰减系数，k越小越尖
    :return: 样本点*权重
    '''
    xMat = np.mat(x)
    yMat = np.mat(y)
    samples_num, dim = np.shape(x)
    weights = np.mat(np.eye(samples_num))
    for i in range(samples_num):
        diffMat = testPoint - xMat[i, :]
        weights[i, i] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xwx = xMat.T * (weights * xMat)
    if np.linalg.det(xwx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    # print('ymat', np.shape(yMat))
    # print('xmat', np.shape(xMat))
    # print('weights', np.shape(weights))
    # print('xwx', np.shape(xwx.I))
    # print('xx', np.shape(weights * yMat))
    w = xwx.I * (xMat.T * (weights * yMat.T))
    return testPoint * w

def lwlrTest(testArr, x, y, k=1.0):
    '''
    局部加权线性回归调用程序
    :param testArr: 需要计算权值的样本点集合
    :param x: x
    :param y: y
    :param k: 控制衰减的系数
    :return: 返回预测值
    '''
    sample_num, dim = np.shape(x)
    y_Hat = np.zeros(sample_num)
    for i in range(sample_num):
        y_Hat[i] = lwlr(testArr[i], x, y, k)
    return y_Hat

def lwlrShowData(x, y, y_Hat):
    '''
    局部加权线性回归的图像显示程序
    :param x: x
    :param y: y
    :param y_Hat: y预测值
    :return: 无，图像显示
    '''
    xMat = np.mat(x)
    srtInd = xMat[:, 1].argsort(0)#返回数组的索引值
    xSort = xMat[srtInd][:, 0, :]
    # print(xSort[:, 1])
    # print(xSort[:, 1].flatten().A[0])
    plt.scatter(xMat[:, 1].flatten().A[0], y, c='g')
    plt.plot(xSort[:, 1], y_Hat[srtInd])
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('图像处理')
    # plt.show()

def ridgeRegress(xMat, yMat, lam=0.2):
    '''
    ridge回归采用矩阵方法求解
    :param xMat:
    :param yMat:
    :param lam:
    :return:
    '''
    # xMat = np.mat(x)
    w = xMat.T * xMat + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(w) == 0.0: #应该不需要进行判断，因为加入lambda后所有的矩阵都可逆,lambda为0的时候仍然需要进行判定
        print('')
        return
    ws = w.I * (xMat.T * yMat)
    return ws

def ridgeDataStandard(x, y):
    '''
    ridge数据标准化
    :param x:
    :param y:
    :return:
    '''
    xMat = np.mat(x)
    yMat = np.mat(y).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMean = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    print('xVar', xVar)
    xMat = (xMat - xMean) / xVar
    numTestPoint  = 30
    wMat = np.zeros((numTestPoint, np.shape(xMat)[1]))
    for i in range(numTestPoint):
        ws = ridgeRegress(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat

def ridgeShowData(wMat):
    '''
    用于ridge回归进行图像展示，主要展示回归系数与log(lambda)的关系，最左边Lambda最小时，所有系数与线性回归的值一致，右边，系数全部缩减为0
    为了定量确定找到最佳的参数值，需要进行交叉验证
    需要判断哪些变量对结果预测最具有影响力，直接观察对应的系数大小
    :param wMat:标准化后的数据
    :return:
    '''
    print(wMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wMat)
    plt.show()

def logisticGradientAscent(x, y, alpha=0.01, max_iter_count=1000):
    n = len(x[0])
    x = np.array(x)
    y = np.array(y)
    iter_count = 0
    weights = []
    for i in range(n):
        weights.append(1.0)
    weights = np.array(weights)
    while iter_count < max_iter_count:
        for i in range(len(x)):
            # print((x[i] * weights).sum())
            y_hat = 1.0 / (1.0 + np.exp(-1 * ((x[i] * weights).sum())))
            weights += alpha * (y[i] - y_hat) * x[i]
        iter_count += 1
    return weights

def logisticGradientAscentMatrix(x, y, alpha=0.01, max_iter_count=1000):
    xMatrix = np.mat(x)
    yMatrix = np.mat(y).transpose()
    m, n = np.shape(xMatrix)
    weights = np.ones((n, 1))
    iter_count = 0
    while iter_count < max_iter_count:
        for i in range(m):
            y_hat = 1.0 / (1.0 + np.exp(-1 * (xMatrix[i] * weights)))
            weights = weights + (alpha * (yMatrix[i] - y_hat) * xMatrix[i]).transpose()
        iter_count += 1
    return weights

if __name__ == '__main__':
    start = time.time()
    # file_path = 'ex0.txt'
    # ridge_datafile_path = 'abalone.txt'
    logistic_dataset_path = 'logisticTestSet.txt'
    # x, y = loadDataSet(file_path) #载入数据
    x, y = loadDataSet(logistic_dataset_path)
    # threat_calc = LinearRegres(x, y)
    # theta_bgd = bgd(np.array(x), np.array(y))
    # theta_other_bgd = Batch_Gradient_Descent(np.array(x), np.array(y))
    # theta_sgd = sgb(np.array(x), np.array(y), alpha=0.0001)
    # theta_mbgd = mbgb(np.array(x), np.array(y), alpha=0.0001)
    # print('纯计算的线性回归解：', threat_calc)
    # print("我的批量梯度下降算法计算的值", theta_bgd)
    # print("我的随机梯度下降算法计算的值", theta_sgd)
    # print("我的随机梯度下降算法计算的值", theta_mbgd)
    # print("其他人最终的批量梯度下降算法计算的值", theta_other_bgd)
    # y_Hat = lwlrTest(x, x, y, k=0.02)#局部加权线性回归
    # wMat = ridgeDataStandard(x, y)
    # ridgeShowData(wMat)
    weights = logisticGradientAscentMatrix(x, y, 0.001, 500)
    print(weights)
    weis = logisticGradientAscent(x, y, alpha=0.001, max_iter_count=500)
    print(weis)
    end = time.time()
    total_time = end - start
    print('运行时间：', total_time)
    # showData(x, y)
    # lwlrShowData(x,y,y_Hat)#局部加权线性回归数据显示
    # showLogisticData(x, y)

