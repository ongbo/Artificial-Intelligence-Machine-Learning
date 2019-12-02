from numpy import *
#从文件中加载数据，元素之间以tab相隔
def loadDataset(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readline():
        curLine = line.strip().split('\t')
        # print(curLine)
        fltLine = map(float, curLine)
        # print(fltLine)
        dataMat.append(fltLine)
    return dataMat

#计算欧氏距离
def distEclud(vecA, vecB):
    return sqrt(num(power(vecA-vecB, 2)))
#对给定的数据集构建一个包含k个随机中心的集合，这里是通过找最小值和最大值，采用0-1随机构造一个处于两者之间的数
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ*random.rand(k,1)
    return centroids
dataset = loadDataset('testSet.txt')
print(dataset)
datamat = mat(dataset)
print(datamat)
