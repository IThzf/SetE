from random import uniform, sample, random
from LPAL import LPAL
from copy import deepcopy
import json
import numpy as np
from torch import nn
import torch
import time

# import SetE
from tqdm import tqdm

def norm(x, pnorm=0):
    if pnorm == 1:
        return torch.sum(torch.abs(x), -1)
    else:
        return torch.sum(x**2,-1)

def normalize_emb(x):
    # return  x/float(length)
    veclen = torch.clamp_min_(torch.norm(x, 2, -1,keepdim=True), 1.0)
    ret = x/veclen
    return ret.detach()

def normalize_radius(x):
    return torch.clamp(x,min=0,max=1.0)


class SetE(nn.Module):
    # 概念、实体、关系三元组初始化
    def __init__(self, conceptList, instanceList, relationList, tripleList, instanceofList, subclassofList, beta = 0.005, learingRate = 0.01, dim = 50, B_t = 1, B_r = 2):
        super(SetE,self).__init__()
        self.learingRate = learingRate
        self.dim = dim
        self.conceptList = conceptList
        self.instanceList = instanceList
        self.instanceList_set = set(instanceList)
        self.instanceofList = instanceofList
        self.subclassofList = subclassofList
        self.relationList = relationList
        self.tripleList = tripleList

        set1 = set()
        for i in range(len(tripleList)):
            set1.add(tuple(tripleList[i]))

        self.tripleList_set = set1

        set1 = set()
        for i in range(len(instanceofList)):
            set1.add(tuple(instanceofList[i]))
        self.instanceof_set = set1
        self.B_t = B_t
        self.B_r = B_r
        self.beta = beta

        self.concept_embeddings = nn.Embedding(len(self.conceptList), self.dim)
        self.instance_embeddings = nn.Embedding(len(self.instanceList), self.dim,max_norm=1.0) # 小于1
        self.rel_embeddings = nn.Embedding(len(self.relationList), 2*self.dim,max_norm=1.0)
        self.criterion = nn.MarginRankingLoss(1, False)
        self.init_weights()
        self.device = "cpu"

    def init_weights(self):
        # nn.init.xavier_uniform(self.concept_embeddings.weight.data)
        # nn.init.xavier_uniform(self.instance_embeddings.weight.data)
        # nn.init.xavier_uniform(self.rel_embeddings.weight.data)

        concept_init = np.random.uniform(0, 6 / np.sqrt(self.dim), (len(self.conceptList), self.dim)) * np.random.uniform(0, 1, (len(self.conceptList), self.dim))
        self.concept_embeddings.weight.data = torch.from_numpy(concept_init)

        instance_init = np.random.uniform(0, 6 / np.sqrt(self.dim), (len(self.instanceList), self.dim)) * np.random.uniform(0, 1, (len(self.instanceList), self.dim))
        self.instance_embeddings.weight.data = torch.from_numpy(instance_init)

        rel_init = np.random.uniform(0, 6 / np.sqrt(2*self.dim), (len(self.relationList), 2*self.dim)) * np.random.uniform(0, 1, (len(self.relationList), 2*self.dim))
        self.rel_embeddings.weight.data = torch.from_numpy(rel_init)

        # self.concept_embeddings.weight.data = 




    # def forward(self,batch_instances,batch_triples):

    def forward_bin(self,batch_pos,batch_neg):

        # batch_instances_pos = batch_instances[0] # 获取正样本
        # batch_instances_neg = batch_instances[1]

        def get_loss_max(b_t, instance,concept, type):

            f_tensor = torch.sum(instance * concept, 1)  # 每个sample相加
            B_t = torch.full(list(f_tensor.shape), b_t) # 构造f_pos.shape维度的tensor，使用self.B_t填充
            if type == 0:
                ans_sub = f_tensor - B_t
            else:
                ans_sub = B_t - f_tensor
            zeros = torch.zeros(list(f_tensor.shape))
            ans_max = torch.stack((ans_sub, zeros), 1)
            f_loss = torch.max(ans_max, 1)[0]

            return f_loss

        losses = None
        pos_flag = 1
        for batchs in (batch_pos,batch_neg):

            batch_instance = torch.from_numpy(batchs[:,0]).to(self.device)
            batch_concept = torch.from_numpy(batchs[:,1]).to(self.device)

            instance = self.instance_embeddings(batch_instance) # 获取instance
            concept = self.concept_embeddings(batch_concept)

            loss = get_loss_max(self.B_t,instance,concept,pos_flag)
            pos_flag -= 1

            if losses == None:
                losses = loss.sum()
            else:
                losses += loss.sum()
            # losses += loss


        # loss = loss.sum()

        return losses

    def forward_tri(self,batch_pos,batch_neg):

        # batch_instances_pos = batch_instances[0] # 获取正样本
        # batch_instances_neg = batch_instances[1]

        def get_loss_max(b_r, heads,tails,rels, type):
            heads_tails = torch.cat((heads,tails),1)
            f_tensor = torch.sum(heads_tails * rels, 1)  # 每个sample相加
            B_r = torch.full(list(f_tensor.shape), b_r) # 构造f_pos.shape维度的tensor，使用self.B_t填充
            if type == 0:
                ans_sub = f_tensor - B_r
            else:
                ans_sub = B_r - f_tensor
            zeros = torch.zeros(list(f_tensor.shape))
            ans_max = torch.stack((ans_sub, zeros), 1)
            f_loss = torch.max(ans_max, 1)[0]

            return f_loss

        losses = None
        pos_flag = 1
        for batchs in (batch_pos,batch_neg):

            batch_head = torch.from_numpy(batchs[:,0]).to(self.device)
            batch_tail = torch.from_numpy(batchs[:,1]).to(self.device)
            batch_rel = torch.from_numpy(batchs[:,2]).to(self.device)

            heads = self.instance_embeddings(batch_head) # 获取instance
            tails = self.instance_embeddings(batch_tail)
            rels = self.rel_embeddings(batch_rel)

            loss = get_loss_max(self.B_r,heads,tails,rels,pos_flag)
            pos_flag -= 1
            # losses += loss
            if losses == None:
                losses = loss.sum()
            else:
                losses += loss.sum()

        # loss = loss.sum(loss)

        return losses


    # 训练过程
    def SetETrain(self, cI=1000):
        lr, rr = self.relationsum()
        print("训练开始")
        optimizer = torch.optim.SGD(self.parameters(),lr=self.learingRate,weight_decay=self.beta)
        self.train()
        self.to(self.device)

        nbatchs = len(self.instanceofList) / 100
        # print(nbatchs)
        # print("第", cycleIndex, "次循环")
        # 二元关系样本获取
        # Sbatchbin = self.getSample(0, 100)
        Sbatchbin = self.instanceofList
        # 三元关系样本获取
        time1 = time.time()
        # Sbatchtri = self.getSample(1, 340000)
        Sbatchtri = self.tripleList
        print("正样本获取时间： ",time.time() - time1)

        for cycleIndex in tqdm(range(1000)):

            Tbatchbin = []
            Tbatchtri = []
            time1 = time.time()
            # 二元、三元负样本生成
            for sbatchbin in Sbatchbin:
                binaryWithCorruptedBinary = [sbatchbin, self.GetNegativeSample(sbatchbin, lr, rr)]  # 存储正负样本的tuple
                # if binaryWithCorruptedBinary not in Tbatchbin:
                Tbatchbin.append(binaryWithCorruptedBinary)


            for sbatchtri in Sbatchtri:
                tripletWithCorruptedTriplet = (sbatchtri, self.GetNegativeSample(sbatchtri, lr, rr))
                # if tripletWithCorruptedTriplet not in Tbatchtri:
                Tbatchtri.append(tripletWithCorruptedTriplet)
            print("负样本生成时间： ", time.time() - time1)


            # 所有instance,relationship满足大于0，小于1

            self.instance_embeddings.weight.data = normalize_radius(self.instance_embeddings.weight.data)

            self.rel_embeddings.weight.data = normalize_radius(self.rel_embeddings.weight.data)


            optimizer.zero_grad()

            # print("type: ",type)
            # Tbatchbin = list(Tbatchbin)
            Tbatchbin = np.array(Tbatchbin)
            Tbatchbin_pos = Tbatchbin[:,0,:]
            Tbatchbin_neg = Tbatchbin[:,1,:]

            # Tbatchbin_tensor_pos = torch.Tensor(Tbatchbin_pos)
            loss_bin = self.forward_bin(Tbatchbin_pos,Tbatchbin_neg)

            Tbatchtri = np.array(Tbatchtri)
            Tbatchtri_pos = Tbatchtri[:, 0, :]
            Tbatchtri_neg = Tbatchtri[:, 1, :]

            # Tbatchbin_tensor_pos = torch.Tensor(Tbatchbin_pos)
            loss_tri = self.forward_tri(Tbatchtri_pos, Tbatchtri_neg)
            loss = loss_bin + loss_tri
            print("loss: ",loss)
            loss.backward()

            # zeros = torch.zeros(list(self.instanceList),self.dim)
            # ans_max = torch.stack((self.instance_embeddings.weight.data, zeros), 1)
            # clip_min_data = torch.max(ans_max, 1)[0]
            # self.instance_embeddings.weight.data = clip_min_data
            #
            # zeros = torch.zeros(list(self.relationList), 2*self.dim)
            # ans_max = torch.stack((self.rel_embeddings.weight.data, zeros), 1)
            # clip_min_data = torch.max(ans_max, 1)[0]
            # self.rel_embeddings.weight.data = clip_min_data



            optimizer.step()

            if cycleIndex % 1 == 0:
                print("进行第%d次循环" % cycleIndex)
                self.write_vector2file("YAGO39K\\Vector\\relationVector.txt",self.rel_embeddings)
                self.write_vector2file("YAGO39K\\Vector\\conceptVector.txt",self.concept_embeddings)
                self.write_vector2file("YAGO39K\\Vector\\instanceVector.txt",self.instance_embeddings)

        # 初始化向量
    def initialize(self):
        conceptVectorList = {}
        instanceVectorList = {}
        relationVectorList = {}
        # 概念向量初始化
        for concept in self.conceptList:
            n = 0
            conceptVectorList[concept] = 1
        print("conceptVector初始化完成，数量是%d" % len(conceptVectorList))

        # 实体向量初始化
        for instance in self.instanceList:
            n = 0
            instanceVectorList[instance] = 1
        print("instanceVector初始化完成，数量是%d" % len(instanceVectorList))

        # 关系向量初始化
        for relation in self.relationList:
            n = 0
            relationVectorList[relation] = 1
        print("relationVectorList初始化完成，数量是%d" % len(relationVectorList))
        self.conceptList = conceptVectorList
        self.instanceList = instanceVectorList
        self.relationList = relationVectorList

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
                while (key_, o, p) in self.tripleList_set:
                    new_s = RAND(self.instanceList)
                    key_ = new_s[0]
                return [key_, new_o, p]
            else:
                new_o = RAND(self.instanceList)
                key_ = new_o[0]
                while (s, key_, p) in self.tripleList_set:
                    new_o = RAND(self.instanceList)
                    key_ = new_o[0]
                return [new_s, key_, p]

        else:
            e, t = sample_[0], sample_[1]
            new_e, new_t = sample_[0], sample_[1]
            new_t = RAND(self.conceptList)
            key_ = new_t[0]
            while (e, key_) in self.instanceof_set:
                new_s = RAND(self.conceptList)
                key_ = new_s[0]
            return [new_e, key_]

    def write_vector2file(self,dir, embeddings):
        embeddings = embeddings.weight.data.numpy().tolist() # embedding装numpy
        with open(dir, 'w', encoding='utf-8') as file:
            for index in range(len(embeddings)):
                file.write(str(index) + "\t")
                file.write(str(embeddings[index]) + "\n")

    def writeConceptVector(self, dir):
        print("写入概念")
        conceptVectorFile = open(dir, 'w', encoding='utf-8')
        for concept in self.conceptList.keys():
            conceptVectorFile.write(str(concept) + "\t")
            conceptVectorFile.write(str(self.conceptList[concept]))
            conceptVectorFile.write("\n")
        conceptVectorFile.close()

    def writeRelationVector(self, dir):
        print("写入关系")
        relationVectorFile = open(dir, 'w', encoding='utf-8')
        for relation in self.relationList.keys():
            relationVectorFile.write(str(relation)+ "\t")
            relationVectorFile.write(str(self.relationList[relation]))
            relationVectorFile.write("\n")
        relationVectorFile.close()

    def writeInstanceVector(self, dir):
        print("写入实体")
        instanceVectorFile = open(dir, 'w', encoding='utf-8')
        for instance in self.instanceList.keys():
            instanceVectorFile.write(str(instance) + "\t")
            instanceVectorFile.write(str(self.instanceList[instance]))
            instanceVectorFile.write("\n")
        instanceVectorFile.close()

    def writeaxiom(self, dir, sp='\t'):
        count = 468
        o = open(dir, 'w', encoding='utf-8')
        for i in range(count):
            l1 = sample(self.conceptList, 1)
            l2 = sample(self.conceptList, 1)
            if i % 100 == 0:
                print(i)
            if l1 == l2:
                continue
            l3 = sample(self.relationList, 1)
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
            list.append(int(DetailsAndId[1]))
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
            # if type == 1:
            #     ins,con = instanceList[int(instance)],conceptList[int(concept)]
            # else:
            #     ins,con = conceptList[int(instance)], conceptList[int(concept)]
            re = [int(instance),int(concept)]
            list.append(re)
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
            # ins1,ins2,rel = instanceList[int(instance1)],instanceList[int(instance2)],relationList[int(res[2])]
            re = [int(instance1),int(instance2),int(relation)]
            list.append(re)
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
    # subclassofNum, subclassofList = openTrainBin(dirInstanceOfTrain,0,instanceList,conceptList)  #此处错误
    subclassofNum, subclassofList = openTrainBin(dirSubclassOfTrain, 0, instanceList, conceptList)
    tripleNum, tripleList = openTrainTri(dirTripleTrain,instanceList,relationList)

    print("打开SetE")
    SetE_ = SetE(conceptList, instanceList, relationList, tripleList, instanceofList, subclassofList, beta = 0.005, learingRate = 0.001, dim = 50, B_t = 1, B_r = 0.5)
    SetE_.initialize()
    SetE_.SetETrain()


    # SetE_.writeInstanceVector("YAGO39K/Vector/instanceVector.txt")
    # SetE_.writeConceptVector("YAGO39K/Vector/conceptVector.txt")
    # SetE_.writeRelationVector("YAGO39K/Vector/relationVector.txt")

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