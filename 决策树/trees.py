# -*- coding: UTF-8 -*-
from math import log
import operator

# 创建数据集
def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        # 如果当前分类不在分类字典中就加入字典
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 书上没有错行，应该写在if外面
    shannonEnt = 0.0
    for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            # 以2为底求对数, 并累差和
            shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 按照给定特征划分数据集
# 参数为：待划分的数据集、划分数据集的特征、特征的返回值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        # 将符合特征的数据抽取出来
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            # 将原列表数据集特征后面的数据提取出来
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的划分方式
# 数据必须是一种由列表组成的列表，所有的列表长度相同
# 数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签
def chooseBestFeatureToSplit(dataSet):
    # 计算数据集的总特征
    numFeatures = len(dataSet[0]) - 1
    # 得到香农熵
    baseEntropy = calsShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        # 计算每种划分方式的信息熵
        newEntropy = 0.0
        for value in uniqueVals:
            # 遍历当前特征中的所有唯一属性值，对每个特征划分一次数据集
            subDataSet = splitDataSet(dataSet, i, value)
            # 然后计算数据集的新熵值
            prob = len(subDataSet) / float(len(dataSet))
            # 并对所有唯一特征值得到的熵求和
            newEntropy = prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # 计算最好的信息熵
        # 返回最好划分的索引值        
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
        print infoGain
    return bestFeature


#
def majorityCnt(classList):
    classCount = {}
    # 创建键值为classList中唯一值得数据字典
    # 字典对象存储了classList中每个类标签出现的频率
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(
        classCount.iteritems(),
        keys=operator.attrgetter(1),
        reverse=True)
    # 返回出现次数最多的分类名称
    return sortedClassCount[0][0]
