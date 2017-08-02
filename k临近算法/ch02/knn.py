# -*- coding: UTF-8 -*-  
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    # 计算欧式距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # axis=1 则按行求和
    # axis=0,则按列求和
    # 不传参数，则结果为全部值加起来
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # argsort()返回的是排序结果的索引index
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选择距离最小的K个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 将文本记录转换为numpy数组
def file2matrix(filename):
    # 打开文件
    fr = open(filename)
    # 读取所有行，每行内容为列表的一个元素
    arrayOLines = fr.readlines()
    # 获取行数
    numberOfLines = len(arrayOLines)
    # 初始化返回矩阵为0
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()     # 除去空格和回车符
        listFromLine = line.split('\t')     # 按制表符分割
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """特征值归一化"""
    minVals = dataSet.min(0)    # 按列计算，从列中选取最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 当前值减去最小值
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 除以范围值，tile可以扩展矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    """测试分类器准确率"""
    # hoRatio = 0.10  # 90%用于训练，10%用于测试
    hoRatio = 0.30
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVlas = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(
            normMat[i, :], normMat[numTestVecs:m, :],
            datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d"\
            % (classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f " % (errorCount / float(numTestVecs))


def classifyPerson():
    """约会网站预测函数"""
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input(
        "percentage of time spent playing video games?"
    ))
    ffMiles = float(raw_input("frequent flier miles earned  per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((
        inArr - minVals) / ranges, normMat, datingLabels, 3)
    print "You will probably like this person: ", \
        resultList[classifierResult - 1]