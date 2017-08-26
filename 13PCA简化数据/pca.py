# coding=utf-8
# author ：Landuy
# time ：2017/8/26
# email ：qq282699766@gamil.com
#####   PCA算法   ####
import numpy as np


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split('\t') for line in fr.readlines()]
    # 构建矩阵
    datArr = [map(float, line) for line in stringArr]
    return np.mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = np.mean(dataMat, axis=0)
    # 去除平均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵
    covMat = np.cov(meanRemoved, rowvar=0)
    # 计算特征值
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # 对特征值进行从小到大的排序
    eigValInd = np.argsort(eigVals)
    # 从小打到对N个值排序
    eigValInd = eigValInd[:-(topNfeat + 1): -1]
    redEigVects = eigVects[:, eigValInd]
    # 将数据转移到新空间
    lowDataMat = meanRemoved * redEigVects
    reconMat = (lowDataMat * redEigVects.T) + meanVals
    return lowDataMat, reconMat


################# 缺失值处理 ####################

# 用NaN替换成平均值的函数
def replaceNanWithMean():
    datMat = loadDataSet('scom.data', ' ')
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        # 计算所有非NaN的平均值
        meanVal = np.mean(datMat[np.nonzero(-np.isnan(datMat[:, i]))[0], i])
        # 将所有的NaN置为平均值
        datMat[np.nonzero(np.isnan(datMat[:, i]))[0], i] = meanVal
    return datMat
