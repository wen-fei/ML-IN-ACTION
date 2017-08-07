# -*-coding: utf-8 -*-
import numpy as np


# SMO算法中的辅助函数
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


# i:是alpha的下标
# m是所有alpha的数目
def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(np.random.uniform(0, m))
    return j


# 用于调整大于H或小于L的值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# 简化版SMO算法
# 参数：数据集，类别标签，常数C,容错率，取消前最大的循环次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    # 没用任何alpha改变的情况下遍历数据集的次数
    iter = 0
    while(iter < maxIter):
        # 用于记录alpha是否已经进行优化
        alphaPairsChanges = 0
        for i in range(m):
            # 我们预测的类别
            fXi = float(
                np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)
            ) + b
            # 误差，预测结果减去真实结果
            Ei = fXi - float(labelMat[i])
            # 如果alpha可以更改，进入优化过程
            if (
                (labelMat[i] * Ei < -toler) 
                and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) \
                and (alphas[i] > 0)
            ):
                j = selectJrand(i, m)
                # 随机选择第二个alpha
                fXj = float(
                    np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)
                ) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 保证alpha在0与C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[j])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print "L==H"
                    continue
                # alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T \
                        - dataMatrix[i, :] * dataMatrix[i, :].T \
                        - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print "eta>=0"
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough"
                    continue
                # 对i进行修改，修改量与j相同，但方向相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) \
                        * dataMatrix[i, :] * dataMatrix[i, :].T \
                        - labelMat[j] * (alphas[j] - alphaJold) \
                        * dataMatrix[i, :] * dataMatrix[j, :].T
                # 设置常数项
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) \
                        * dataMatrix[i, :] * dataMatrix[j, :].T \
                        - labelMat[j] * (alphas[j] - alphaJold) \
                        * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanges += 1
                print "iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanges)
        if (alphaPairsChanges == 0):
            iter += 1
        else:
            iter = 0
        print "iteration number : %d " % iter
    return b, alphas

