from random import uniform, sample, random
from LPAL import LPAL
from copy import deepcopy
import json
import numpy as np
import time
class SetE:
    # 概念、实体、关系三元组初始化
    def __init__(self, conceptList, instanceList, relationList, tripleList, instanceofList, subclassofList, beta = 0.005, learingRate = 0.01, dim = 50, B_t = 1, B_r = 2):
        self.learingRate = learingRate
        self.dim = dim
        self.conceptList = conceptList
        self.instanceList = instanceList
        self.instanceofList = instanceofList
        self.subclassofList = subclassofList
        self.relationList = relationList
        self.tripleList = tripleList
        self.B_t = B_t
        self.B_r = B_r
        self.beta = beta

    # 初始化向量
    def initialize(self):
        conceptVectorList = {}
        instanceVectorList = {}
        relationVectorList = {}
        # 概念向量初始化
        for concept in self.conceptList:
            n = 0
            conceptVector = []
            while n < self.dim:
                ram = init(0, self.dim)  # 初始化的范围
                conceptVector.append(ram)
                n += 1
            conceptVector = norm(conceptVector)  # 归一化
            conceptVectorList[concept] = conceptVector
        print("conceptVector初始化完成，数量是%d" % len(conceptVectorList))

        # 实体向量初始化
        for instance in self.instanceList:
            n = 0
            instanceVector = []
            while n < self.dim:
                ram = init(0,self.dim)
                instanceVector.append(ram)
                n += 1
            instanceVector = norm(instanceVector)  # 归一化
            instanceVectorList[instance] = instanceVector
        print("instanceVector初始化完成，数量是%d" % len(instanceVectorList))

        # 关系向量初始化
        for relation in self.relationList:
            n = 0
            relationVector = []
            while n < (self.dim * 2):
                ram = init(1,self.dim)
                relationVector.append(ram)
                n += 1
            relationVector = norm(relationVector)  # 归一化
            relationVectorList[relation] = relationVector
        print("relationVectorList初始化完成，数量是%d" % len(relationVectorList))
        self.conceptList = conceptVectorList
        self.instanceList = instanceVectorList
        self.relationList = relationVectorList

    # 训练过程
    def SetETrain(self, cI=1000):
        lr, rr = self.relationsum()
        print("训练开始")

        Sbatchbin = self.getSample(0, 1000)
        # 三元关系样本获取
        Sbatchtri = self.getSample(1, 1000)
        Tbatchbin = []
        Tbatchtri = []
        # 二元、三元负样本生成
        for sbatchbin in Sbatchbin:
            binaryWithCorruptedBinary = (sbatchbin, self.GetNegativeSample(sbatchbin, lr, rr))  # 存储正负样本的tuple
            if binaryWithCorruptedBinary not in Tbatchbin:
                Tbatchbin.append(binaryWithCorruptedBinary)

        for sbatchtri in Sbatchtri:
            tripletWithCorruptedTriplet = (sbatchtri, self.GetNegativeSample(sbatchtri, lr, rr))
            if tripletWithCorruptedTriplet not in Tbatchtri:
                Tbatchtri.append(tripletWithCorruptedTriplet)


        for cycleIndex in range(cI):

            print("第", cycleIndex, "次循环")
            # 二元关系样本获取
            time1 = time.time()
            self.update(Tbatchbin, Tbatchtri)
            print(time.time()-time1)
            if cycleIndex % 501 == 0:
                print("进行第%d次循环" % cycleIndex)
                self.writeRelationVector("YAGO39K\\Vector\\relationVector.txt")
                self.writeConceptVector("YAGO39K\\Vector\\conceptVector.txt")
                self.writeInstanceVector("YAGO39K\\Vector\\instanceVector.txt")
                
    # 统计关系覆盖头尾实体个数
    def relationsum(self):
        relationleft = {}
        relationright = {}
        for i in self.relationList.keys():
            relationleft[i] = 0
            relationright[i] = 0
        for i in self.relationList.keys():
            for j in self.tripleList:
                if j[2] == i:
                    relationleft[i] += 1
                    relationright[i] += 1
        return relationleft, relationright

    # 获取二元、三元小样本
    def getSample(self, tag, size):
        if tag == 0:
            return sample(self.instanceofList, size)
        else:
            return sample(self.tripleList, size)

    # 生成二元、三元负样本
    def GetNegativeSample(self, sample_, lr, rr):
        if len(sample_) == 3:
           s, o, p = sample_[0], sample_[1], sample_[2]
           new_s, new_o = sample_[0], sample_[1]
           r_n = random()
           rn = lr[p] / (lr[p] + rr[p])
           if r_n < rn:
              new_s = RAND(self.instanceList)
              key_ = new_s[0]
              while (key_, o, p) in self.tripleList:
                    new_s = RAND(self.instanceList)
                    key_ = new_s[0]
              return (key_, new_o, p)
           else:
              new_o = RAND(self.instanceList)
              key_ = new_o[0]
              while (s, key_, p) in self.tripleList:
                     new_o = RAND(self.instanceList)
                     key_ = new_o[0]
              return (new_s, key_, p)

        else:
             e,t = sample_[0],sample_[1]
             new_e,new_t = sample_[0],sample_[1]
             new_t = RAND(self.conceptList)
             key_ = new_t[0]
             while (e, key_) in self.instanceofList:
                    new_s = RAND(self.conceptList)
                    key_ = new_s[0]
             return (new_e, key_)

    # 二元、三元关系更新
    def update(self, Tbatchbin, Tbatchtri):
        copyConceptList = deepcopy(self.conceptList)
        copyInstanceList = deepcopy(self.instanceList)
        copyRelationList = deepcopy(self.relationList)

        # 二元关系训练
        for binaryWithCorruptedbinary in Tbatchbin:
            headInstanceVector = copyInstanceList[binaryWithCorruptedbinary[0][0]] # 正样本
            tailConceptVector = copyConceptList[binaryWithCorruptedbinary[0][1]]
            headInstanceVectorWithCorrupted = copyInstanceList[binaryWithCorruptedbinary[1][0]] # 负样本
            tailConceptVectorWithCorrupted = copyConceptList[binaryWithCorruptedbinary[1][1]]

            # 样本训练前
            headInstanceVectorBeforeBatch = self.instanceList[binaryWithCorruptedbinary[0][0]]
            tailConceptVectorBeforeBatch = self.conceptList[binaryWithCorruptedbinary[0][1]]
            headInstanceVectorWithCorruptedBeforeBatch = self.instanceList[binaryWithCorruptedbinary[1][0]]
            tailConceptVectorWithCorruptedBeforeBatch = self.conceptList[binaryWithCorruptedbinary[1][1]]

            sumPos = BinaryDistance(headInstanceVectorBeforeBatch,tailConceptVectorBeforeBatch)
            sumNeg = BinaryDistance(headInstanceVectorWithCorruptedBeforeBatch,tailConceptVectorBeforeBatch)

            if sumPos < self.B_t:
                s = self.B_t - sumPos
                # 二元关系样本向量更新
                # 正样本
                tempPositiveInstanceb = subVector([self.learingRate * 2 * s * i for i in tailConceptVectorBeforeBatch],[self.learingRate * self.beta * i for i in headInstanceVectorBeforeBatch])
                tempPositiveConcept = subVector([self.learingRate * 2 * 0.1 * i for i in headInstanceVectorBeforeBatch],[self.learingRate * self.beta * 0.1 * i for i in tailConceptVectorBeforeBatch])

                # 正样本头尾实体进行更新
                headInstanceVector = addVector(headInstanceVector, tempPositiveInstanceb)
                tailConceptVector = addVector(tailConceptVector, tempPositiveConcept)

            if sumNeg >= self.B_t:
                s = sumNeg - self.B_t
                s = 0 - s
                tempNegtativeInstanceb = subVector([self.learingRate * 2 * s * i for i in tailConceptVectorWithCorruptedBeforeBatch],[self.learingRate * self.beta * i for i in headInstanceVectorWithCorruptedBeforeBatch])
                tempNegtativeConcept = subVector([self.learingRate * 2 * 0.1 * i for i in headInstanceVectorWithCorruptedBeforeBatch],[self.learingRate * self.beta * 0.1 * i for i in tailConceptVectorWithCorruptedBeforeBatch])

                # 负样本头尾实体更新
                headInstanceVectorWithCorrupted = subVector(headInstanceVectorWithCorrupted,tempNegtativeInstanceb)
                tailConceptVectorWithCorrupted = subVector(tailConceptVectorWithCorrupted,tempNegtativeConcept)

                # 只归一化这几个刚更新的向量
                copyInstanceList[binaryWithCorruptedbinary[0][0]] = norm(headInstanceVector)
                copyConceptList[binaryWithCorruptedbinary[0][1]] = norm(tailConceptVector)
                copyInstanceList[binaryWithCorruptedbinary[1][0]] = norm(headInstanceVectorWithCorrupted)
                copyConceptList[binaryWithCorruptedbinary[1][1]] = norm(tailConceptVectorWithCorrupted)

        # 三元关系训练
        for tripleWithCorruptedtriple in Tbatchtri:
            headInstance1Vector = copyInstanceList[tripleWithCorruptedtriple[0][0]]  # 正样本
            tailInstance2Vector = copyInstanceList[tripleWithCorruptedtriple[0][1]]
            relationVector = copyRelationList[tripleWithCorruptedtriple[0][2]]
            headInstance1VectorWithCorrupted = copyInstanceList[tripleWithCorruptedtriple[1][0]]  # 负样本
            tailInstance2VectorWithCorrupted = copyInstanceList[tripleWithCorruptedtriple[1][1]]

            # 样本训练前
            headInstance1VectorBeforeBatch = self.instanceList[tripleWithCorruptedtriple[0][0]]
            tailInstance2VectorBeforeBatch = self.instanceList[tripleWithCorruptedtriple[0][1]]
            relationVectorBeforeBatch = self.relationList[tripleWithCorruptedtriple[0][2]]
            headInstance1VectorWithCorruptedBeforeBatch = self.instanceList[tripleWithCorruptedtriple[1][0]]
            tailInstance2VectorWithCorruptedBeforeBatch = self.instanceList[tripleWithCorruptedtriple[1][1]]

            sum1Pos = TripletDistance(headInstance1VectorBeforeBatch,relationVectorBeforeBatch,tailInstance2VectorBeforeBatch)
            sum1Neg = TripletDistance(headInstance1VectorWithCorruptedBeforeBatch,relationVectorBeforeBatch,tailInstance2VectorWithCorruptedBeforeBatch)

            # 三元关系样本向量更新
            # 实体更新
            if sum1Pos < self.B_r:
                s = self.B_r - sum1Pos
                tempPositiveInstance1 = subVector([self.learingRate * 2 * s * i for i in relationVectorBeforeBatch[:50]],[self.learingRate * self.beta * i for i in headInstance1VectorBeforeBatch])
                tempPositiveInstance2 = subVector([self.learingRate * 2 * s * i for i in relationVectorBeforeBatch[50:100]],[self.learingRate * self.beta * i for i in tailInstance2VectorBeforeBatch])
                relationleft = subVector([self.learingRate * self.beta * i for i in headInstance1VectorBeforeBatch],[self.learingRate * 2 * s * i for i in relationVectorBeforeBatch[:50]])
                relationright = subVector([self.learingRate * self.beta * i for i in tailInstance2VectorBeforeBatch],[self.learingRate * 2 * s * i for i in relationVectorBeforeBatch[50:100]])

                # 正样本头尾实体进行更新
                headInstance1Vector = addVector(headInstance1Vector, tempPositiveInstance1)
                tailInstance2Vector = addVector(tailInstance2Vector,tempPositiveInstance2)

                # 关系更新
                relationVector[:50] = addVector(relationVector[:50], relationleft)
                relationVector[50:100] = addVector(relationVector[50:100], relationright)

            if sum1Neg >= self.B_r:
                s = sum1Neg - self.B_r
                s = 0 - s
                tempNegtativeInstance1 = subVector([self.learingRate * 2 * s * i for i in relationVectorBeforeBatch[:50]],[self.learingRate * self.beta * i for i in headInstance1VectorWithCorruptedBeforeBatch])
                tempNegtativeInstance2 = subVector([self.learingRate * 2 * s * i for i in relationVectorBeforeBatch[50:100]],[self.learingRate * self.beta * i for i in tailInstance2VectorWithCorruptedBeforeBatch])
                relationleft = subVector([self.learingRate * self.beta * i for i in headInstance1VectorWithCorruptedBeforeBatch],[self.learingRate * 2 * s * i for i in relationVectorBeforeBatch[:50]])
                relationright = subVector([self.learingRate * self.beta * i for i in tailInstance2VectorWithCorruptedBeforeBatch],[self.learingRate * 2 * s * i for i in relationVectorBeforeBatch[50:100]])
                # 关系更新
                relationVector[:50] = addVector(relationVector[:50], relationleft)
                relationVector[50:100] = addVector(relationVector[50:100], relationright)

                # 负样本头尾实体更新
                headInstance1VectorWithCorrupted = subVector(headInstance1VectorWithCorrupted,tempNegtativeInstance1)
                tailInstance2VectorWithCorrupted = subVector(tailInstance2VectorWithCorrupted,tempNegtativeInstance2)

            # 只归一化这几个刚更新的向量
            copyInstanceList[tripleWithCorruptedtriple[0][0]] = norm(headInstance1Vector)
            copyInstanceList[tripleWithCorruptedtriple[0][1]] = norm(tailInstance2Vector)
            copyRelationList[tripleWithCorruptedtriple[0][2]] = norm(relationVector)
            copyInstanceList[tripleWithCorruptedtriple[1][0]] = norm(headInstance1VectorWithCorrupted)
            copyInstanceList[tripleWithCorruptedtriple[1][1]] = norm(tailInstance2VectorWithCorrupted)


        self.conceptList = copyConceptList
        self.instanceList = copyInstanceList
        self.relationList = copyRelationList


    def writeConceptVector(self, dir):
        print("写入概念")
        conceptVectorFile = open(dir, 'w', encoding='utf-8')
        for concept in self.conceptList.keys():
            conceptVectorFile.write(concept + "\t")
            conceptVectorFile.write(str(self.conceptList[concept]))
            conceptVectorFile.write("\n")
        conceptVectorFile.close()

    def writeRelationVector(self, dir):
        print("写入关系")
        relationVectorFile = open(dir, 'w', encoding='utf-8')
        for relation in self.relationList.keys():
            relationVectorFile.write(relation + "\t")
            relationVectorFile.write(str(self.relationList[relation]))
            relationVectorFile.write("\n")
        relationVectorFile.close()

    def writeInstanceVector(self, dir):
        print("写入实体")
        instanceVectorFile = open(dir, 'w', encoding='utf-8')
        for instance in self.instanceList.keys():
            instanceVectorFile.write(instance + "\t")
            instanceVectorFile.write(str(self.instanceList[instance]))
            instanceVectorFile.write("\n")
        instanceVectorFile.close()

    def writeaxiom(self,dir, sp='\t'):
        count = 468
        o = open(dir, 'w', encoding='utf-8')
        for i in range(count):
            l1 = sample(self.conceptList, 1)
            l2 = sample(self.conceptList,1)
            if i % 100 == 0:
                print(i)
            if l1 == l2:
                continue
            l3 = sample(self.relationList,1)
            o.write(l3[0] + '\t')
            o.write(l1[0] + '\t')
            o.write(l2[0] + '\n')
        o.close()

# 初始化概念、实体、关系向量
def init(type, dim):
    if type == 0:
       return uniform(-6 / (dim ** 0.5), 6 / (dim ** 0.5)) * uniform(0,1)
    else:
       return uniform(-6 / ((2 * dim) ** 0.5), 6 / ((2 * dim) ** 0.5)) * uniform(0,1)


# 二元关系得分函数,e,t为向量
def BinaryDistance(e, t):
    res = 0
    for i in range(len(e)):
        res += e[i] * t[i]
    return res


# 三元关系得分函数,s,p,o为向量
def TripletDistance(s, p, o):
    t = deepcopy(s)
    t.extend(o)
    res = 0
    for i in range(len(p)):
        res += t[i] * p[i]
    return res


# 余弦相似度
def Cosinesimilarity(s1, s2):
    sum_ = temp1 = temp2 = 0
    for i in range(len(s1)):
        sum_ += s1[i] * s2[i]
        temp1 += s1[i] ** 2
        temp2 += s2[i] ** 2
    res = sum_ / ((temp1 ** 0.5) * (temp2 ** 0.5))
    return res


# L2正则项
def distanceL2(v):
    res = 0
    for i in range(len(v)):
        res += v[i] ** 2
    return res ** 0.5


# 向量相加
def addVector(a, b):
    res = []
    for i in range(len(a)):
        res.append(a[i] + b[i])
    return res


# 向量相减
def subVector(a, b):
    res = []
    for i in range(len(a)):
        res.append(a[i] - b[i])
    return res


# 随机选出实体
def RAND(list):
    return sample(list.keys(), 1)


# 对向量进行归一化
def norm(list):
    for i in range(len(list)):
        if list[i] < 0:
            list[i] = 0
        elif list[i] > 1:
            list[i] = 1
    x = distanceL2(list)
    if x > 1:
       for i in range(len(list)):
           list[i] /= x
    return list







# 读取概念、实体、关系id
def openDetailsAndId(dir, sp="\t"):
    idNum = 0
    list = []
    with open(dir, encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            DetailsAndId = line.strip().split(sp)
            list.append(DetailsAndId[0])
            idNum += 1
    return idNum, list


# 读取二元关系组,type表示二元关系类型
def openTrainBin(dir, type, instanceList, conceptList, sp="\t"):
    num = 0
    list = []
    with open(dir, encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            temp = triple[0]
            begin = 0
            res = []
            for i in range(len(temp)):
                if temp[i] == " ":
                    res.append(temp[begin:i])
                    begin = i + 1
            res.append(temp[begin:])
            if len(res) <= 1:
                continue
            instance,concept = res[0],res[1]
            if type == 1:
               ins,con = instanceList[int(instance)],conceptList[int(concept)]
            else:
               ins,con = conceptList[int(instance)], conceptList[int(concept)]
            re = [ins,con]
            list.append(tuple(re))
            num += 1
        return num, list


# 读取三元关系组
def openTrainTri(dir, instanceList, relationList, sp="\t"):
    num = 0
    list = []
    with open(dir,encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            temp = triple[0]
            begin = 0
            res = []
            for i in range(len(temp)):
                if temp[i] == " ":
                    res.append(temp[begin:i])
                    begin = i + 1
            res.append(temp[begin:])
            if len(res) <= 1:
                continue
            instance1,instance2,relation = res[0],res[1],res[2]
            ins1,ins2,rel = instanceList[int(instance1)],instanceList[int(instance2)],relationList[int(res[2])]
            re = [ins1,ins2,rel]
            list.append(tuple(re))
            num += 1
        return num, list

# 读取向量
def openVector(dir, sp="\t"):
    list = {}
    with open(dir,encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            nameandvector = line.strip().split(sp)
            list[nameandvector[0]] = json.loads(nameandvector[1])
    return list

def openTrain(dir,sp="\t"):
    num = 0
    list = []
    with open(dir,encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if(len(triple)<3):
                continue
            list.append(tuple(triple))
            num += 1
    return num, list

if __name__ == '__main__':
    dirInstance = "YAGO39K/Train/instance2id.txt"
    instanceIdNum, instanceList = openDetailsAndId(dirInstance)
    dirConcept = "YAGO39K/Train/concept2id.txt"
    conceptIdNum, conceptList = openDetailsAndId(dirConcept)
    dirRelation = "YAGO39K/Train/relation2id.txt"
    relationIdNum, relationList = openDetailsAndId(dirRelation)
    dirInstanceOfTrain = "YAGO39K/Train/instanceOf2id.txt"
    dirSubclassOfTrain = "YAGO39K/Train/subClassOf2id.txt"
    dirTripleTrain = "YAGO39K/Train/triple2id.txt"
    instanceofNum, instanceofList = openTrainBin(dirInstanceOfTrain,1,instanceList,conceptList)
    subclassofNum, subclassofList = openTrainBin(dirInstanceOfTrain,0,instanceList,conceptList)  #此处错误
    # subclassofNum, subclassofList = openTrainBin(dirSubclassOfTrain, 0, instanceList, conceptList)
    tripleNum, tripleList = openTrainTri(dirTripleTrain,instanceList,relationList)

    print("打开SetE")
    SetE_ = SetE(conceptList, instanceList, relationList, tripleList, instanceofList, subclassofList, beta = 0.005, learingRate = 0.001, dim = 50, B_t = 1, B_r = 0.5)
    SetE_.writeaxiom('YAGO39K/Axioms/objectsomevaluesfrom.txt')
    print("SetE初始化")
    SetE_.initialize()
    SetE_.SetETrain()
    SetE_.writeInstanceVector("YAGO39K/Vector/instanceVector.txt")
    SetE_.writeConceptVector("YAGO39K/Vector/conceptVector.txt")
    SetE_.writeRelationVector("YAGO39K/Vector/relationVector.txt")

    # 读取向量
    conceptList_ = openVector("YAGO39K/Vector/conceptVector.txt")
    instanceList_ = openVector("YAGO39K/Vector/instanceVector.txt")
    relationList_ = openVector("YAGO39K/Vector/relationVector.txt")
    print("开始公理学习")
    LPAL_ = LPAL(subclassofList, instanceofList, tripleList, conceptList_, instanceList_, relationList_, MinSC_t = 1, MinSC_r = 0.8, MinHC = 1, LB_t = 1, LB_r = 0.5)
    print("开始学习subclass公理")
    subclassof_ = LPAL_.SubClassOf()
    LPAL_.writeAxioms(subclassof_, "YAGO39K/Axioms/subclassof.txt")
    print("开始学习subproperty公理")
    subproperty_ = LPAL_.SubpropertyOf()
    LPAL_.writeAxioms(subproperty_, "YAGO39K/Axioms/subpropertyof.txt")
    print("开始学习objectvalues公理")
    objectsomevaluesfrom_ = LPAL_.ObjectSomeValuesFrom()
    LPAL_.writeAxioms(objectsomevaluesfrom_, "YAGO39K/Axioms/objectsomevaluesfrom.txt")