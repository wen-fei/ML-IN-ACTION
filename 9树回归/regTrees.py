# coding=utf-8
# author ：Landuy
# time ：2017/8/23
# email ：qq282699766@gamil.com
import numpy as np


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # 将每行映射成浮点数
        dataMat.append(fltLine)
    return dataMat


# 二分
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :] # 代码错误  去掉结尾的[0]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# 建立叶节点
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


# 误差估计函数
def regErr(dataSet):
    # 总方差 = 均方差 * 数据样本点个数
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


# 回归树的切分函数
# @ops： 用户提供的参数
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 容许的误差下降值
    tolS = ops[0]
    # 切分的最少样本数
    tolN = ops[1]
    # 如果所有值相等则退出
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        # 代码错误 原本是dataSet[:, featIndex],报错TypeError: unhashable type: 'matrix'解决方法
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if(np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果误差减小不大，则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分出的数据集很小则退出
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


# 创建树
# @leafType: 建立叶节点的函数
# @errType: 误差计算函数
# @ops: 树构建所需其他参数的元组
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 递归出口, 返回叶节点值
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # 左子树和右子树
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 递归创建左子树和右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# 回归树剪枝函数
# 判断是不是一个树（非叶子）
def isTree(obj):
    return (type(obj).__name__ == 'dict')


# 对树进行塌陷处理，即返回平均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


# 剪枝
# @tree: 待剪枝的树
# @testData： 剪枝所需要的测试数据
def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right'])) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        if isTree(tree['left']):
            tree['left'] = prune(tree['left'], lSet)
        if isTree(tree['right']):
            tree['right'] = prune(tree['right'], rSet)
    # 不再是子树
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 不合并误差
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
            np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 合并误差
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        # 合并的误差比不合并的误差小
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree


#### 模型树 ######

# 模型树的叶节点生成函数
def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    # 将X与Y中的数据格式化
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError("This matrix is singular , cannot do inverse \n " + \
                        "try increasing the second value of ops")
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


# 当数据不需要再切分的时候生成叶节点的模型
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    # 返回回归系数
    return ws


# 在给定的数据集上计算误差
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    # 返回yHat与Y之间的平方误差
    return np.sum(np.power(Y - yHat, 2))



########################## 树回归与标准回归的比较 #####################
# 用树回归进行预测的代码
# 为了与函数modelTreeEval()保持一致，保留两个输入参数
def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inData):
    n = np.shape(inData)[1]
    X = np.mat(np.ones((1, n+1)))
    X[:, 1:n+1] = inData
    return float(X * model)


def treeForceCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForceCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)

    else:
         if isTree(tree['right']):
             return treeForceCast(tree['right'], inData, modelEval)
         else:
             return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForceCast(tree, np.mat(testData[i]), modelEval)
    return yHat




















