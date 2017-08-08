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
                    # print "L==H"
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


# 完整版Platt SMO 的支持函数
class optStruct:
    # kTup: 包含核函数信息的元祖
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # 误差缓存
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

    # 计算E值 
    def calcEk(self, oS, k):
        # fXk = float(np.multiply(oS.alphas, oS.labelMat).T *
                    # (oS.X * oS.X[k, :].T)) + oS.b
        # Ek = fXk - float(oS.labelMat[k])
        # 使用核函数所做的修改
        fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
        Ek = fXk - float(oS.labelMat[k])
        return Ek

    # 选择第二个alpha
    def selectJ(self, i, oS, Ei):
        # 内循环中的启发式方法
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        oS.eCache[i] = [1, Ei]
        validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
        if len(validEcacheList) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.calcEk(oS, k)
                deltaE = abs(Ei - Ek)
                # 选择具有最大步长的j
                if (deltaE > maxDeltaE):
                    maxX = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = selectJrand(i, oS.m)
            Ej = self.calcEk(oS, j)
        return j, Ej

    # 将误差存入缓存中
    def updateEk(self, oS, k):
        Ek = self.calcEk(oS, k)
        oS.eCache[k] = [1, Ek]

    # 完整Platt SMO算法中的优化历程
    def innerL(self, i, oS):
        Ei = self.calcEk(oS, i)
        if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
                ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
            j, Ej = self.selectJ(i, oS, Ei)
            alphaIold = oS.alphas[i].copy()
            alphaJold = oS.alphas[i].copy()
            if (oS.labelMat[i] != oS.labelMat[j]):
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L == H:
                # print "L == H"
                return 0
            
            # eta = 2.0 * oS.X[i, :] * oS.X[j, :].T \
            #         - oS.X[i, :] * oS.X[i, :].T \
            #         - oS.X[j, :] * oS.X[j, :].T
            # 为了使用核函数所做修改
            eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
            if eta >= 0:
                print "eta >=0 "
                return 0
            oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
            oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
            self.updateEk(oS, j)
            if (abs(oS.alphas[j] - alphaJold) < 0.00001):
                print "j not moving enough"
                return 0
            oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] \
                                * (alphaJold - oS.alphas[j])
            self.updateEk(oS, j)
            # b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) \
            #         * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] \
            #         * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
            # b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) \
            #         * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] \
            #         * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
            # 为了使用核函数所做的修改
            b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
                    oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * \
                    oS.K[i, j]
            b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
                    oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * \
                    oS.K[j, j]
            if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
                oS.b = b1
            elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
                oS.B = b2
            else:
                oS.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0


# 完整版Platt SMO外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entrieSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entrieSet)):
        alphaPairsChanged = 0
        if entrieSet:
            for i in range(oS.m):
                alphaPairsChanged += oS.innerL(i, oS)
            print "fullSet, iter : %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += oS.innerL(i, oS)
                print "non-bound, iter: %d i :%d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        if entrieSet:
            entrieSet = False
        elif (alphaPairsChanged == 0):
            entrieSet = True
        print "iteration number: %d" % iter
    return oS.b, oS.alphas


# 
def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


# 高斯核函数
def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


# 利用核函数进行分类的径向基测试函数
def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    # 构建支持向量矩阵
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print "there are %d Support Vectors " % np.shape(sVs)[0]
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print "the training error rate is: %f" % (float(errorCount) / m)
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print "the test error rate is: %f" % (float(errorCount) / m)


# 基于SVM的手写数字识别
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' %(dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rng', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print "there are %d Support Vectors " % np.shape(sVs)[0]
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print "the training error rate is: %f" % (float(errorCount) / m)
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print "the test error rate is: %f" % (float(errorCount) / m)


