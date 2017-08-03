# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt


# 使用文本注解绘制树节点

# 定义文本框和箭头样式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 绘制带箭头的注解
# 参数含义 nodeText : 文本内容
# contenrPt ： 注解坐标
# parentPt ： 箭头起始坐标
def plotNode(nodeText, conterPt, parentPt, nodeType):
    createPlot.ax1.annotate(
        nodeText,
        xy=parentPt,
        xycoords='axes fraction',
        xytext=conterPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    # 清空绘图区
    fig.clf()
    # createPlot.ax1 = plt.subplot(111, frameon=False)
    # 绘制两个代表不同类型的树节点
    # plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    # plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)

    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))        # 树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))       # 树的深度
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


# 获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试节点的数据类型是否为字典
        # 如果是一个字典，说明该节点也是一个判断节点，需要递归
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 获取树的层数，计算遍历过程中遇到判断节点的个数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 预先存储树的信息
def retrieveTree(i):
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {
            'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]


# 画出树结构
def plotMidText(cntrPt, parentPt, txtString):
    # 在父子节点间填充信息
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    # 计算宽与高
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (
        plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
        plotTree.yOff
    )
    # 标记子节点属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 减少y偏移
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(
                secondDict[key], (plotTree.xOff, plotTree.yOff), 
                cntrPt, leafNode
            )
            plotMidText(
                (plotTree.xOff, plotTree.yOff), cntrPt, str(key)
            )
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


