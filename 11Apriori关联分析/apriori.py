# coding=utf-8
# author ：Landuy
# time ：2017/8/25
# email ：qq282699766@gamil.com


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 构建C1，C1是大小为1的所有候选项的集合
# 然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求
# 那些满足最低要求的项集构成L1，而L1中的元素相互构成C2， C2再进一步过滤变成L2.。。。
def createC1(dataSet):
    C1 = []
    # 遍历每一个记录
    for transaction in dataSet:
        # 遍历记录中的每一个项
        for item in transaction:
            if not [item] in C1:
                # 添加只包含该物品项的一个列表
                # 因为python不能创建只有一个整数的集合
                C1.append([item])
    C1.sort()
    # 对C1中的每个项构建一个不变集合
    # frozenset类型，指被冰冻，不可改变
    # C1是一个集合的集合{{1}，{2}...{N}}
    return map(frozenset, C1)


# 用于从C1生成L1
# @Ck：数据集
# @D：包含候选集合的列表
# @minSupport：最小支持度
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                # 如果C1中的集合在记录中已经存在，则+1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    # 计算支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems
        # 如果支持度满足最小支持度的要求
        if support >= minSupport:
            # 在列表的首部插入任意新的集合
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


############ Apriori算法##########
# 创建数据集Ck（候选集列表）
# @Lk：频繁项集列表Lk
# @K：项集元素个数
def aprioriGen(Lk, k): # creates ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            # 前k-2个项相同时，将两个集合合并
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        # 扫描数据集，从Ck到Lk
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


######## 关联规则生成函数 ##########
# @L： 频繁项集列表
# @supportData: 包含那些频繁项集支持数据的字典
# @minconf：最小可信度阈值
def generateRules(L, supportData, minconf=0.7):
    bigRuleList = []
    # 只获取有两个或更多元素的集合
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                relusFromConseq(freqSet, H1, supportData, bigRuleList, minconf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minconf)
    return bigRuleList


# 对规则进行评估
def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print freqSet - conseq, '-->', conseq, 'conf:', conf
            br1.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


# 生成候选规则集合
# @freqSet：频繁项集
# @H: 可以出现在规则右部的元素列表H
def relusFromConseq(freqSet, H, supportData, bigRuleList, minconf):
    # H中的频繁集大小
    m = len(H[0])
    # 尝试进一步合并
    if (len(freqSet) > (m+1) ):
        # 创建Hm+1条新候选规则
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, bigRuleList, minconf)
        if (len(Hmp1) > 1):
            relusFromConseq(freqSet, Hmp1, supportData, bigRuleList, minconf)


################# 发现国会投票中的模式 ###############

from time import sleep
# from votesmart import votesmart
# votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
# votesmart.votes.getBillsByStateRecent()
# 无法申请到key，放弃




























