# -*- coding:utf-8 -*-
#__author__: 'Baojian Jiang'
#__time__:'2018/12/18'

# adaboost

import numpy as np

def loadData():
    datMat = np.matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

