# coding=utf-8
import numpy as np


# 导入数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 标准回归函数
def standRegress(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    # np.linalg 是linear algebra 缩写，线性代数包
    # np.linalg.det(xTx) 返回的是xTx的行列式
    if np.linalg.det(xTx) == 0.0:
        print "this is singular, connot do inverse "
        return
    # 回归系数
    ws = xTx.I * (xMat.T * yMat)
    return ws


# 局部甲醛线型回归函数
# k 控制衰减速度
def lwlr(testPoint, xArr, yArr, k=1.0):
    # 将列表形式的数据转为numpy矩阵形式
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 样本数量
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m))) # 创建对角矩阵，为每个样本初始化一个权重
    for j in range(m):
        # 权重值大小以指数级衰减
        # 计算预测点与该样本的偏差
        diffMat = testPoint - xMat[j, :]
        # 根据偏差利用高斯核函数赋予该样本相应的权重
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    # 求矩阵的内积
    xTx = xMat.T * (weights * xMat)
    print "xTx is :", xTx, "\n"
    if np.linalg.det(xTx) == 0.0:
        print "this Matrix is singlar, cannot do inverse"
        return
    # 根据公式计算回归系数
    ws = xTx.I * (xMat.T * (weights * yMat))
    # 计算测试点的预测值
    return testPoint * ws


#
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


# 预测鲍鱼的年龄
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


# 岭回归
# 计算回归系数
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

#
def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 数据标准化，使每纬特征具有相同的重要性
    # 所有特征减去各自的均值并除以方差
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat


def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)   #calc mean then subtract it off
    inVar = np.var(inMat, 0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans) / inVar
    return inMat

# 前向逐步线型回归
# @eps 每次迭代需要调整的步长
# @ numIt 迭代次数
def stageWise(xArr, yArr, eps=0.01, numIt = 100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat
















