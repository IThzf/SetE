from numpy import *
import SetE
import json
from copy import copy
from scipy import optimize as op
import numpy as np

# 链接预测
class Test:
    def __init__(self, conceptList, relationList,  instanceList,  subclassofListTest, instanceofListTest, tripleListTest, label = "head", B_t = 1, B_r = 2):
        self.conceptList = conceptList
        self.instanceList = instanceList
        self.relationList = relationList
        self.subclassofListTest = subclassofListTest
        self.instanceofListTest = instanceofListTest
        self.tripleListTest = tripleListTest
        self.rank = []
        self.label = label
        self.B_t = B_t
        self.B_r = B_r
        self.LB_t = 1

    def writeRank(self, dir):
        print("写入rank")
        file = open(dir, 'w', encoding = 'utf-8')
        for r in self.rank:
            file.write(str(r[0]) + "\t")
            file.write(str(r[1]) + "\t")
            file.write(str(r[2]) + "\t")
            file.write('\n')
        file.close()

    # 三元关系实体预测
    def getTriRank(self):
        count_ = 0
        print("迭代总数：", len(self.tripleListTest) * len(self.instanceList))
        for triplet in self.tripleListTest:

            rankList = {}
            for instanceTemp in self.instanceList.keys():
                count_ += 1
                if count_ % 100 == 0:
                    print(count_)
                if self.label == "head":
                    rankList[instanceTemp] = self.Tridis(self.instanceList[instanceTemp], self.instanceList[triplet[1]], self.relationList[triplet[2]])
                else:
                    # 替换尾实体
                    rankList[instanceTemp] = self.Tridis(self.instanceList[triplet[0]], self.instanceList[instanceTemp], self.relationList[triplet[2]])
                # 得分从小到大排列
                nameRank = sorted(rankList.items(), key = lambda y: y[1])
                if self.label == 'head':
                    numTri = 0
                else:
                    numTri = 1
                # 记录正确答案位置
                x = 1
                for i in nameRank:
                    if i[0] == triplet[numTri]:
                        break
                    x += 1
                # 记录每个三元关系正确答案的预测排名
                self.rank.append((triplet, triplet[numTri], x))

    # 二元关系预测,type表示二元关系类型
    def getBinRank(self,type):
        if type == 0:
            count_ = 0
            print("迭代总数：", len(self.subclassofListTest) * len(self.conceptList))

            count_out = 0
            for subclassof in self.subclassofListTest:
                rankList = {}
                count_out += 1
                print("count_out: ",count_out)
                if count_out >= 10:
                    break
                count_in = 0
                for conceptTemp in self.conceptList.keys():

                    count_in += 1
                    if count_in % 1000 == 0:
                        print("count_in: ",count_in)
                    if self.label == "head":
                        # rankList[conceptTemp] = self.Bindis(self.conceptList[conceptTemp], self.conceptList[subclassof[1]])
                        rankList[conceptTemp] = self.isSubClassOf(conceptTemp, subclassof[1])
                    else:
                        # rankList[conceptTemp] = self.Bindis(self.conceptList[subclassof[0]], self.conceptList[conceptTemp])
                        rankList[conceptTemp] = self.isSubClassOf(subclassof[0], conceptTemp)
                    # 升序排列
                nameRank = sorted(rankList.items(), key = lambda y: y[1])
                if self.label == 'head':
                    numBin = 0
                else:
                    numBin = 1
                x = 1
                for i in nameRank:
                    if i[0] == subclassof[numBin]:
                        break
                    x += 1
                print(x)
                self.rank.append((subclassof, subclassof[numBin], x))
        else:
            print("迭代总数：", len(self.instanceofListTest))
            count_out = 0
            for instanceof in self.instanceofListTest:
                rankList = {}
                if self.label == "head":

                    count_out += 1
                    if count_out == 100:
                        break
                    print("count_out_instanceOf:",count_out)

                    count_in = 0
                    for instanceTemp in self.instanceList.keys():
                        count_in += 1
                        # if count_in % 1000 == 0:
                        #     print("count_in: ", count_in)
                        rankList[instanceTemp] = self.Bindis(self.instanceList[instanceTemp],self.conceptList[instanceof[1]])

                else:
                    for conceptTemp in self.conceptList.keys():
                        rankList[conceptTemp] = self.Bindis(self.instanceList[instanceof[0]],self.conceptList[conceptTemp])

                # 升序排列
                nameRank = sorted(rankList.items(), key = lambda y: y[1])
                if self.label == 'head':
                    numBin = 0
                else:
                    numBin = 1
                x = 1
                for i in nameRank:
                    if i[0] == instanceof[numBin]:
                        break
                    x += 1
                self.rank.append((instanceof, instanceof[numBin], x))

    # 关系预测
    def getRelationRank(self):
        self.rank = []
        for triplet in self.tripleListTest:
            rankList = {}
            for relationTemp in self.relationList.keys():
                if self.relationList[relationTemp] == []:
                    continue
                rankList[relationTemp] = self.Tridis(self.instanceList[triplet[0]], self.instanceList[triplet[1]], self.relationList[relationTemp])
            nameRank = sorted(rankList.items(), key = lambda y: y[1])
            x = 1
            for i in nameRank:
                if i[0] == triplet[2]:
                    break
                x += 1
            self.rank.append((triplet, triplet[2],  x))

    # MRR
    def getMeanRank(self):
        num = 0
        for r in self.rank:
            num += 1 / r[2]
        return num / len(self.rank)

    # Hits@N
    def getHitsN(self, N):
        print("rank: ",self.rank)
        num = 0
        for r in self.rank:
            if r[2] <= N:
                num += 1
        return num / len(self.rank)

    # 三元关系距离函数
    def Tridis(self, h, t, r):
        res = 0
        temp = copy(h)
        temp.extend(t)
        for i in range(len(temp)):
            res += temp[i] * r[i]
        return self.B_r - res

    # 二元关系距离函数
    def Bindis(self, e, t):
        # print("len: " ,len(e))
        res = 0
        res = np.dot(e,t)

        # for i in range(len(e)):
        #     res += e[i] * t[i]

        return res


    def NegativeVector(self,a):
        for i in range(len(a)):
            a[i] = - a[i]
        return a
    # 向量比较
    def Compare(self,a, b):
        res = 0
        for i in range(len(a)):
            if a[i] <= b[i]:
                res += 1
        return res

    def isSubClassOf(self,c_1,c_2):

        return self.LinearP1(c_1, c_2)

        # if self.Compare(self.conceptList[c_1], self.conceptList[c_2]) == 50:
        #     return 100
        #
        # else:
        #     return self.LinearP1(c_1, c_2)
        #
        #
        # return 0

    # SubClassOf
    # 线性规划函数
    def LinearP1(self, C_i, C_j):
        # 线性规划x的向量维度为概念向量维度
        ConceptList = self.conceptList
        tC_i,tC_j = ConceptList[C_i],ConceptList[C_j]
        # 目标函数
        c = np.array(tC_j)
        # 约束条件
        # T,B分别表示Top,Bottom的类型向量表示
        T = [1] * 50
        B = [0] * 50
        A_ub = np.array([self.NegativeVector(T), B, tC_i])
        B_ub = np.array([-self.LB_t, self.LB_t, -self.LB_t])
        # x的范围
        x = (0,1)
        res = op.linprog(c,A_ub,B_ub,bounds=x)
        return res.fun



# 读取关系
def openDtri(dir, sp="\t"):
    num = 0
    list = []
    with open(dir, encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            data_ = line.strip().split(sp)
            list.append(tuple(data_))
            num += 1
    return num, list

# 加载向量文件
def openVector(dir, sp="\t"):
    list = {}
    with open(dir,encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            nameandvector = line.strip().split(sp)
            list[int(nameandvector[0])] = json.loads(nameandvector[1])
    return list

if __name__ == '__main__':
    dirInstance = "YAGO39K\\Train\\instance2id.txt"
    instanceIdNum, instanceList_ = SetE.openDetailsAndId(dirInstance)
    dirConcept = "YAGO39K\\Train\\concept2id.txt"
    conceptIdNum, conceptList_ = SetE.openDetailsAndId(dirConcept)
    dirRelation = "YAGO39K\\Train\\relation2id.txt"
    relationIdNum, relationList_ = SetE.openDetailsAndId(dirRelation)
    # 读取向量
    dirConceptVector = "YAGO39K\\Vector\\conceptVector.txt"
    conceptList = openVector(dirConceptVector)
    dirInstanceVector = "YAGO39K\\Vector\\instanceVector.txt"
    instanceList = openVector(dirInstanceVector)
    dirRelationVector = "YAGO39K\\Vector\\relationVector.txt"
    relationList = openVector(dirRelationVector)
    # 读取测试集
    dirtriTest = "YAGO39K\\Test\\triple2id_positive.txt"
    dirsubclassofTest = "YAGO39K\\Test\\subClassOf2id_positive.txt"
    dirinstanceofTest = "YAGO39K\\Test\\instanceOf2id_positive.txt"
    tripleNumTest, tripleListTest = SetE.openTrainTri(dirtriTest, instanceList_, relationList_ )
    subclassofNumTest,subclassofListTest = SetE.openTrainBin(dirsubclassofTest, 0, instanceList_, conceptList_)
    instanceofNumTest,instanceofListTest = SetE.openTrainBin(dirinstanceofTest, 1, instanceList_, conceptList_)

    print("开始测试")
    # 对二元关系、三元关系头概念、实体以及关系进行预测
    testHead = Test(conceptList, relationList, instanceList, subclassofListTest, instanceofListTest, tripleListTest, label = 'head', B_t = 1, B_r = 0.5)
    # # 对三元组实体进行预测
    # testHead.getTriRank()
    # testHead.writeRank("YAGO39K\\Prediction\\tripleinstanceRankhead.txt")
    # print(testHead.getMeanRank(),testHead.getHitsN(10))
    # testHead.rank.clear()
    # # 对subclassof关系进行预测
    # print("subclassof关系预测")
    # testHead.getBinRank(0)
    # testHead.writeRank("YAGO39K\\Prediction\\subclassofRankhead.txt")
    # print(testHead.getMeanRank(), testHead.getHitsN(10))
    # testHead.rank.clear()
    # 对instanceof关系进行预测
    testHead.getBinRank(1)
    testHead.writeRank("YAGO39K\\Prediction\\instanceofRankhead.txt")
    print(testHead.getMeanRank(), testHead.getHitsN(10))
    testHead.rank.clear()
    # # 对三元组关系进行预测
    # testHead.getRelationRank()
    # testHead.writeRank("YAGO39K\\Prediction\\triplerelationRankhead.txt")
    # print(testHead.getMeanRank(), testHead.getHitsN(10))
    # testHead.rank.clear()

    # 对尾概念、实体进行预测
    testTail = Test(conceptList, relationList,  instanceList,  subclassofListTest, instanceofListTest, tripleListTest, label = "tail", B_t = 1, B_r = 0.5)
    # 对三元组实体进行预测
    # testTail.getTriRank()
    # testTail.writeRank("YAGO39K\\Prediction\\tripleinstanceRanktail.txt")
    # print(testTail.getMeanRank(), testTail.getHitsN(10))
    # testTail.rank.clear()
    # 对subclassof关系进行预测
    # testTail.getBinRank(0)
    # testTail.writeRank("YAGO39K\\Prediction\\subclassofRanktail.txt")
    # print(testTail.getMeanRank(), testTail.getHitsN(10))
    # testTail.rank.clear()
    # 对instanceof关系进行预测
    testTail.getBinRank(1)
    testTail.writeRank("YAGO39K\\Prediction\\instanceofRanktail.txt")
    print(testTail.getMeanRank(), testTail.getHitsN(10))
    testTail.rank.clear()
    # # 对三元组关系进行预测
    # testTail.getRelationRank()
    # testTail.writeRank("YAGO39K\\Prediction\\triplerelationRanktail.txt")
    # print(testTail.getMeanRank(), testTail.getHitsN(10))
    # testTail.rank.clear()





