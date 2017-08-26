# coding=utf-8
# author ：Landuy
# time ：2017/8/26
# email ：qq282699766@gamil.com


class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}      # 存放节点的子节点

    def inc(self, numOccur):
        self.count += numOccur

    # 将树以文本形式显示
    def disp(self, ind=1):
        print ' ' * ind, self.name, '  ', self.count
        for child in self.children.values():
            child.disp(ind + 1)


# FP树构建函数
def createTree(dataSet, minSup=1):
    headerTable = {}
    # 第一遍扫描数据集并统计每个元素项出现的频度
    # 信息存储在头指针表中
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 扫描头指针表
    # 移除不满足最小支持度的元素项
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    # 如果没有元素项满足（所有项都不频繁）要求就退出
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # 创建根节点
    retTree = treeNode('Null Set', 1, None)
    # 再一次遍历数据集，只考虑频繁项，进行排序
    for tranSet, count in dataSet.items():
        localD = {}
        # 根据全局频率对每个事物中的元素进行排序
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(
                localD.items(),
                key=lambda p: p[1],
                reverse=True
            )]
            # 使用排序后的频率项集对树进行填充
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable


# 让树生长
def updateTree(items, inTree, headerTable, count):
    # 测试第一个元素是否作为子节点存在
    # 如果存在更新元素项计数
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    # 如果不存在则创建一个新的treeNode并将其作为一个子节点添加到树中
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            # 更新头指针表
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])

    # 对剩下的元素迭代调用updateTree函数
    if len(items) > 1:
        # 不断调用自身，每次调用会去掉列表中的第一个元素
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


########## 简单数据集及数据包装器 ############
def loadSimpleDat():
    simpleData = [
        ['r', 'z', 'h', 'j', 'p'],
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
        ['z'],
        ['r', 'x', 'n', 'o', 's'],
        ['y', 'r', 'x', 'z', 'q', 't', 'p'],
        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
    ]
    return simpleData


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


################ 抽取条件模式基 ##################
# 条件模式基石以所查找元素项为结尾的路径集合。每一条路径都是一条前缀路径
# 简单来说，一条前缀路径是介于所查找元素项和树根节点之间的所有内容

# 为给定元素项生成一个条件模式基

# 发现以给定元素项结尾的所有路径的函数
def ascendTree(leafNode, prefixPath):
    # 迭代上述整棵树
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    # 条件模式基字典
    condPats = {}
    while treeNode != None:
        prefixPath = []
        # 上溯FP树
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

########### 创建条件FP树 ############
# 递归查找频繁项集的mineTree函数
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # 从头指针表的底端开始，按出现频率进行排序
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1])]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        # 添加到频繁项集列表
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # 从条件模式基来构建条件FP树
        myCondTree, myHead = createTree(condPattBases, minSup)

        # 挖掘条件FP树
        # 如果树中有元素项的话
        if myHead != None:
            print 'conditional tree for :', newFreqSet
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)















