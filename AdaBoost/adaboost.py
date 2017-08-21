# -*-coding: utf-8 -*-
import numpy as np


def loadSimpData():
    dataMat = np.matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


# 单层决策树生成函数
# 通过阈值比较对数据进行分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = 1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    # 用于存储给定权重向量D时所得到的最佳单层决策树相关信息
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    # 初始化为无穷大
    minError = np.inf
    # 在数据集的所有特征上遍历
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        # 在求出步长以后按段遍历
        for j in range(-1, int(numSteps) + 1):
            # 在大于和小于之间切换不等式
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                # 返回分类预测结果
                predictedVals = stumpClassify(
                    dataMatrix, i, threshVal, inequal
                    )
                # 错误向量
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                print "split : dim %d, thresh %2.f, thresh \
                    inequal: %s, the weighted error is %.3f" \
                    % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = 1
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    # 返回字典，最小错误率和类别估计值
    return bestStump, minError, bestClasEst


# 基于单层决策树的AdaBoost训练过程
# numIt 迭代次数
def adaBoostTrainsDS(dataArr, classLabels, numIt=4.0):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.ones((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print "D:", D.T
        # 该变量告诉总分类器本次单层决策树输出结果的权重
        # max(error, e - 16) 用于确保没有错误时不会发生除零溢出
        alpha = float(0.5 * np.log((1.0 - error) / max(error, np.e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst: ", classEst.T
        # 为下一次迭代计算D
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 错误率累加计算
        aggClassEst += alpha * classEst
        print "aggClassEst: ", aggClassEst.T
        aggErrors = np.multiply(
            np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1))
            )
        errorRate = aggErrors.sum() / m
        print "total error : ", errorRate, "\n"
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


# AdaBoost分类函数
# 利用多个弱分类器进行分类
# datToClass : 待分类样例
# classifierArr : 多个弱分类器组成的数组
def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    # 待分类样例个数
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(
            dataMatrix, classifierArr[i]['dim'],
            classifierArr[i]['thresh'],
            classifierArr[i]['ineq']
            )
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    # 返回aggClassEst的符号，大于0则 返回+1，小于0返回-1
    return np.sign(aggClassEst)


# 自适应数据加载函数
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# ROC曲线的绘制以及AUC计算函数
# predStrengths : 分类器的预测强度
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    # 保留绘制光标的位置
    cur = (1.0, 1.0)
    # 计算AUC的值
    ySum = 0.0
    # 正例的数目
    numPosClas = np.sum(np.array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    # 获取排序索引
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print "the Area Under the Curve is :", ySum * xStep
