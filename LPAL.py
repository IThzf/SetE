from scipy import optimize as op
import numpy as np

# 线性规划公理学习器
class LPAL:
    # LB_t,LB_r,MinSC_t,MinSC_r,MinHC分别为类型、关系集合边界以及各类公理最小置信度
    def __init__(self, subclassofList, instanceofList, tripleList, conceptList, instanceList, relationList, MinSC_t, MinSC_r, MinHC, LB_t = 1, LB_r= 2):
        self.subclassofList = subclassofList
        self.instanceofList = instanceofList
        self.tripleList = tripleList
        self.conceptList = conceptList
        self.instanceList = instanceList
        self.relationList = relationList
        self.LB_t = LB_t
        self.LB_r = LB_r
        self.MinSC_t = MinSC_t
        self.MinSC_r = MinSC_r
        self.MinHC = MinHC

    # 学习subclassof公理
    def SubClassOf(self):
        candidate = []
        count = 0
        # 遍历所有概念
        for i in self.conceptList.keys():
            for j in self.conceptList.keys():
                if i == j:
                   continue
                count += 1
                if count % 100 == 0:
                    print("已学习",count,sep="")
                # 若j概念在每一维度上的值都大于i概念,则应放入subclassof公理候选集
                if Compare(self.conceptList[i],self.conceptList[j]) == 50:
                   if [i,j] not in candidate:
                      candidate.append([i,j])
                elif self.LinearP1(i,j) >= self.LB_t:
                     if [i,j] not in candidate:
                        candidate.append([i,j])
                if count >= 10000:
                    break
            print(count)

        res = self.Filterbin(0,candidate)
        return res

    # 学习SubPropertyOf公理
    def SubpropertyOf(self):
        candidate = []
        count = 0
        # 遍历所有关系
        for i in self.relationList.keys():
            for j in self.relationList.keys():
                if i == j:
                   continue
                count += 1
                if count % 100 == 0:
                   print("已学习",count,sep="")
                if Compare(self.relationList[i], self.relationList[j]) == 100:
                    if [i, j] not in candidate:
                       candidate.append([i, j])
                elif self.LinearP2(i, j) >= self.LB_r:
                     if [i, j] not in candidate:
                        candidate.append([i, j])
        res = self.Filterbin(1,candidate)
        return res

    # 学习SubClassOf(ObjectSomeValuesFrom(Pi,Cj),Ck)公理
    def ObjectSomeValuesFrom(self):
        candidate = []
        count = 0
        # 遍历所有概念、关系
        for j in self.conceptList.keys():
            for k in self.conceptList.keys():
                if j == k:
                   continue
                for i in self.relationList.keys():
                    if self.relationList[i] == []:
                        continue
                    count += 1
                    if count % 100 == 0:
                       print("已学习",count,sep="")
                    if self.LinearP3(i,j,k) >= self.LB_t:
                       if [i,j,k] not in candidate:
                          candidate.append([i,j,k])
            if count >= 1000000:
                break
        res = self.Filtertri(candidate)
        return res

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
        A_ub = np.array([NegativeVector(T), B, tC_i])
        B_ub = np.array([-self.LB_t, self.LB_t, -self.LB_t])
        # x的范围
        x = (0,1)
        res = op.linprog(c,A_ub,B_ub,bounds=x)
        return res.fun

    # SubPropertyOf
    # 线性规划函数
    def LinearP2(self, P_i, P_j):
        RelationList = self.relationList
        rP_i,rP_j = RelationList[P_i],RelationList[P_j]
        # 与relation向量维度一致
        T = [1] * 100
        B = [0] * 100
        c = np.array(rP_j)
        A_ub = np.array([NegativeVector(T),B,NegativeVector(T),B,NegativeVector(rP_i)])
        B_ub = np.array([-self.LB_t,self.LB_t,-self.LB_r,self.LB_r,-self.LB_r])
        x = (0,1)
        res = op.linprog(c, A_ub, B_ub, bounds=x)
        return res.fun

    # SubClassOf(ObjectSomeValuesFrom(Pi,Cj),Ck)
    # 线性规划函数
    def LinearP3(self, P_i, C_j, C_k):
        ConceptList = self.conceptList
        RelationList = self.relationList
        rP_i,tC_j,tC_k = RelationList[P_i],ConceptList[C_j],ConceptList[C_k]
        # 与relation向量维度一致
        T = [1] * 50
        B = [0] * 50
        c = np.array(tC_k + B)
        A_ub = np.array([NegativeVector(T + B), NegativeVector(B + T), NegativeVector(rP_i), NegativeVector(B + tC_j)])
        B_ub = np.array([-self.LB_t, -self.LB_t, -self.LB_r, -self.LB_t])
        x = (0,1)
        res = op.linprog(c, A_ub, B_ub, bounds=x)
        return res.fun

    # 获取某条subclassof公理的标准置信度
    def SC_t(self, C_i, C_j):
        C_ie = C_je = 0
        temp = []
        # 遍历instanceofList中的概念
        for i in self.instanceofList:
            if i[1] == C_i:
               C_ie += 1
               # 记录对应的实体
               temp.append(i[0])
            if i[1] == C_j:
               C_je += 1
               temp.append(i[0])
        C_ije = len(temp) - len(set(temp))
        SC_t = C_ije / C_ie
        return SC_t

    # 获取某条subpropertyof公理的置信度
    def SC_r(self, C_i, C_j):
        C_ir = C_jr = 0
        temp = []
        # 遍历tripleList中的三元关系
        for i in self.tripleList:
            if i[2] == C_i:
               C_ir += 1
               # 记录对应的实体对tuple
               temp.append((i[0],i[1]))
            if i[2] == C_j:
               C_jr += 1
               temp.append((i[0],i[1]))
        C_ijr = len(temp) - len(set(temp))
        SC_r = C_ijr / C_ir
        return SC_r

    # 获取某条SubClassOf(ObjectSomeValuesFrom(Pi,Cj),Ck)公理置信度
    def HC(self, P_i, C_j, C_k):
        # 对于instanoflist中与Cj有关系,且存在Pi关系的实体,同时也是Ck的实体
        C_je = C_jke = 0
        temp = []
        for i in self.instanceofList:
            if i[1] == C_j:
               for j in self.tripleList:
                   if j[2] == P_i:
                      if j[0] == i[0] or j[1] == i[0]:
                         C_je += 1
                         temp.append(i[0])
                         for k in self.instanceofList:
                             if k[0] == i[0] and k[1] == C_k:
                                C_jke += 1
        HC = C_jke / C_je
        return HC

    # list为二元关系集合,type为list类型
    # 筛选函数
    def Filterbin(self, type, list):
        res = []
        # subclassof公理
        if type == 0:
           for i in list:
               e,t = i[0],i[1]
               SC = self.SC_t(e,t)
               if SC > self.MinSC_t:
                  res.append([e,t])
        # subpropertyof公理
        else:
            for i in list:
                p1,p2 = i[0],i[1]
                SC = self.SC_r(p1,p2)
                if SC > self.MinSC_r:
                   res.append([p1,p2])
        return res

    # list为三元关系集合
    # 筛选函数
    def Filtertri(self, list):
        res = []
        for i in list:
            pi,ej,ek = i[0],i[1],i[2]
            HC = self.HC(pi,ej,ek)
            if HC > self.MinHC:
                res.append([pi,ej,ek])
        return res


    def writeAxioms(self, list, dir):
        print("写入公理")
        AxiomsFile = open(dir, 'w', encoding='utf-8')
        for axioms in list:
            if len(axioms) == 2:
               AxiomsFile.write(axioms[0] + "\t" + axioms[1])
               AxiomsFile.write("\n")
            else:
               AxiomsFile.write(axioms[0] + "\t" + axioms[1] + "\t" + axioms[2])
               AxiomsFile.write("\n")
        AxiomsFile.close()

# 向量比较
def Compare(a, b):
    res = 0
    for i in range(len(a)):
        if a[i] <= b[i]:
            res += 1
    return res


# 向量取反
def NegativeVector(a):
    for i in range(len(a)):
        a[i] = - a[i]
    return a


def Cosinesimilarity(s1, s2):
    sum_ = temp1 = temp2 = 0
    for i in range(len(s1)):
        sum_ += s1[i] * s2[i]
        temp1 += s1[i] ** 2
        temp2 += s2[i] ** 2
    res = sum_ / ((temp1 ** 0.5) * (temp2 ** 0.5))
    return res
