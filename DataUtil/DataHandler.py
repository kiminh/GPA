# -*- coding: utf-8 -*-
# @Time    : 2019/12/9 21:33
# @Author  : zxl
# @FileName: DataHandler.py

import numpy as np

def LoadData():

    root = "C://zxl/Data/GPA-large/"
    train_root = root + "train/"
    test_root = root + "test/"
    trainX_path = train_root + "trainX2.npy"
    trainY_path = train_root + "trainY2.npy"
    testX_path = test_root + "testX2.npy"
    testY_path = test_root + "testY2.npy"

    train_X = np.load(trainX_path)
    train_y = np.load(trainY_path)
    test_X = np.load(testX_path)
    test_y = np.load(testY_path)
    return (train_X, train_y, test_X, test_y)