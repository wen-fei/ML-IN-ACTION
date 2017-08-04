# -*- coding:utf-8 -*-
import numpy as np
import operator
import feedparser


# 词表到向量的转换函数
def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字，0代表正常言论
    return postingList, classVec


# 创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])
    # 创建两个集合的并集
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# vocabList:词汇表
# inputSet：某个文档
# 输出：文档向量
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个和词汇表等长的全为0的向量
    returnVec = [0] * len(vocabList)
    # 遍历文档中所有单词
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec


# 朴素贝叶斯分类器训练函数
# trainMatrix: 文档矩阵
# trainCategory: 每篇文档类别标签所构成的向量
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 如果其中一个概率为0，则最后结果也为0，所以讲所有词出现的次数初始化为1，并将分母初始化为2
    # p0Num = np.zeros(numWords)
    # p1Num = np.zeros(numWords)
    # p0Denom = 0.0
    # p1Denom = 0.0
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 某个词在某一文档中出现
        if trainCategory[i] == 1:
            # 改词对应的个数加1
            p1Num += trainMatrix[i]
            # 在所有的文档中，该文档的总词数相应加1
            p1Denom += np.sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    # 对每个元素除以该类别中的总词数
    # p1Vect = p1Num / p1Denom
    # p0Vect = p0Num / p0Denom
    # 
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # np 对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第二个元素相乘
    # 将词汇表中所有词的对应值相加，然后将该值加到类别的对数概率上
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)     # 二分类才可以1减去
    # 返回大概率对应的类别标签
    if p1 > p0:
        return 1
    else:
        return 0


# 便利函数 convenience function
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for positionDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, positionDoc))

    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)


# 朴素贝叶斯词袋模型
def bagOfWord2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 文本解析
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    # 出去类似py、en这样的小词以及空格组成的字符
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# 使用朴素贝叶斯进行交叉验证
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    # 随机选择10封邮件为测试集
    # 随机构建训练集
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        # 选择出的数字所对应的文档被添加到测试集，同时也从训练集中剔除
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    # 遍历测试集，对其中每一封电子邮件进行分类
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print "the error rate is :", float(errorCount) / len(testSet)


# RSS源分类器
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    # 遍历词汇表中的每个词，并统计他在文本中出现的次数
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    # 根据出现次数从高到底对词典排序
    sortedFreq = sorted(
        freqDict.iteritems(),
        key=operator.itemgetter(1),
        reverse=True
    )
    return sortedFreq[:10]


# 高频词去除函数
def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen)
    testSet = []
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWord2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWord2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print "the error rate is :", float(errorCount) / len(testSet)
    return vocabList, p0V, p1V


# 最具表征性词汇显示函数
def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(
        topSF, key=lambda pair: pair[1],
        reverse=True
    )
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(
        topNY, key=lambda pair: pair[1],
        reverse=True
    )
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]