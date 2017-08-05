# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 


# Logistic 回归梯度上升优化算法
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


# dataMatIn 二维numpy数组，每列分别代表不同的特征，每行则代表每个训练样本
# dataMatIn是100X3的矩阵
def gradAscent(dataMatIn, classLabels):
    # 转换为Numpy矩阵数据类型
    dataMatrix = np.mat(dataMatIn)
    # 行向量转化为列向量：转置
    labelMat = np.mat(classLabels).transpose()
    # 得到矩阵大小参数m行n列
    m, n = np.shape(dataMatrix)
    # 步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        # 矩阵相乘，h是一个列向量
        h = sigmoid(dataMatrix * weights)
        # 计算真实类别与预测类别的差值
        error = (labelMat - h)
        # 按照差值方向调整回归系数
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# 画出数据集和Logistic回归最佳拟合直线
def plotBestFit(wei):
    weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcords1 = []
    ycords1 = []
    xcords2 = []
    ycords2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcords1.append(dataArr[i, 1])
            ycords1.append(dataArr[i, 2])
        else:
            xcords2.append(dataArr[i, 1])
            ycords2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcords1, ycords1, s=30, c='red', marker='s')
    ax.scatter(xcords2, ycords2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    # 最佳拟合曲线
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 随机梯度上升算法
def stocGradAscent(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(np.sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 改进的随机梯度上身算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            # alpha 会随着迭代次数不断减小，但不会减小到0
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机选取更新
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(np.sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


