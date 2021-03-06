import SetE
import json
from copy import copy
from Link_Prediction2 import Test
import numpy as np
from scipy import optimize as op
class Classification:
    def __init__(self,conceptlist,instancelist,relationlist,instanceofpos,instanceofneg,subclassofpos,subclassofneg,triplepos,tripleneg,B_t,B_r):

        self.conceptlist = conceptlist
        self.instancelist = instancelist
        self.relationlist = relationlist
        self.subclassofpos = subclassofpos
        self.subclassofneg = subclassofneg
        self.instanceofpos = instanceofpos
        self.instanceofneg = instanceofneg
        self.triplepos = triplepos
        self.tripleneg = tripleneg
        self.B_t = B_t
        self.B_r = B_r
        self.LB_t = 1

    def Tripleclassification(self):
        poslist,neglist = [],[]
        sum_p,sum_n = 0,0
        for (s,p,o) in self.triplepos:
            sum_p += self.Tridis(self.instancelist[s],self.instancelist[p],self.relationlist[o])
            if self.Tridis(self.instancelist[s],self.instancelist[p],self.relationlist[o]) > 0:
                poslist.append((s,p,o))
            else:
                neglist.append((s,p,o))

        for (s,p,o) in self.tripleneg:
            sum_n += self.Tridis(self.instancelist[s],self.instancelist[p],self.relationlist[o])
            if self.Tridis(self.instancelist[s],self.instancelist[p],self.relationlist[o]) > 0:
                poslist.append((s,p,o))
            else:
                neglist.append(((s,p,o)))


        TP,FP = self.get_TP(1,poslist,self.triplepos)
        TN,FN = self.get_TN(1,neglist,self.tripleneg)
        a,b,c,d = self.Accuracy(TP,FP,TN,FN)
        return a,b,c,d

    def Instanceofclassification(self):
        poslist, neglist = [], []
        for (s,p) in self.instanceofpos:
            if self.Bindis(self.instancelist[s], self.conceptlist[p]) > 0:
                poslist.append((s, p))
            else:
                neglist.append((s, p))
        for (s, p) in self.instanceofneg:
            if self.Bindis(self.instancelist[s], self.conceptlist[p]) > 0:
                poslist.append((s, p))
            else:
                neglist.append((s, p))

        TP, FP = self.get_TP(0, poslist, self.instanceofpos)
        TN, FN = self.get_TN(0, neglist, self.instanceofneg)
        a, b, c, d = self.Accuracy(TP, FP, TN, FN)
        return a,b,c,d

    def Subclassofclassification(self):
        poslist, neglist = [], []
        count_ = 0
        for (s, p) in self.subclassofpos:
            if self.isSubClassOf(s,p):
                poslist.append((s, p))
            else:
                neglist.append((s, p))
        for (s, p) in self.subclassofneg:
            if self.isSubClassOf(s,p) > 0:
                poslist.append((s, p))
            else:
                neglist.append((s, p))

        TP, FP = self.get_TP(0, poslist, self.subclassofpos)
        TN, FN = self.get_TN(0, neglist, self.subclassofneg)
        a, b, c, d = self.Accuracy(TP, FP, TN, FN)
        return a,b,c,d

    # poslist,neglist?????????????????????????????????,list??????????????????
    def get_TP(self, type, poslist, list):
        TP,FP = 0,0
        if type == 0:
           for pos in poslist:
               s,p = pos[0],pos[1]
               if [s,p] in list:
                   TP += 1
               else:
                   FP += 1
        else:
           for pos in poslist:
               s, p, o = pos[0], pos[1], pos[2]
               if (s, p, o) in list:
                   TP += 1
               else:
                   FP += 1
        return TP,FP

    def get_TN(self, type, neglist, list):
        TN, FN = 0, 0
        if type == 0:
            for neg in neglist:
                s, p = neg[0], neg[1]
                if [s,p] in list:
                    TN += 1
                else:
                    FN += 1
        else:
            for neg in neglist:
                s, p ,o = neg[0], neg[1] ,neg[2]
                if (s, p, o) in list:
                    TN += 1
                else:
                    FN += 1
        return TN, FN

    def Accuracy(self, TP, FP, TN, FN):
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        return  accuracy, precision, recall, F1

    # ????????????????????????
    def Tridis(self, h, t, r):
            res = 0
            temp = copy(h)
            temp.extend(t)
            for i in range(len(temp)):
                res += temp[i] * r[i]
            return res - self.B_r

    # ????????????????????????
    def Bindis(self, e, t):
            res = 0
            for i in range(len(e)):
                res += e[i] * t[i]
            return res - self.B_t

    def NegativeVector(self,a):
        for i in range(len(a)):
            a[i] = - a[i]
        return a
    # ????????????
    def Compare(self,a, b):
        res = 0
        for i in range(len(a)):
            if a[i] <= b[i]:
                res += 1
        return res

    def isSubClassOf(self,c_1,c_2):
        if self.Compare(self.conceptlist[c_1], self.conceptlist[c_2]) == 50:
            return True

        if self.LinearP1(c_1, c_2) >= self.LB_t:
            return True

        return False

    # SubClassOf
    # ??????????????????
    def LinearP1(self, C_i, C_j):
        # ????????????x????????????????????????????????????
        ConceptList = self.conceptlist
        tC_i,tC_j = ConceptList[C_i],ConceptList[C_j]
        # ????????????
        c = np.array(tC_j)
        # ????????????
        # T,B????????????Top,Bottom?????????????????????
        T = [1] * 50
        B = [0] * 50
        A_ub = np.array([self.NegativeVector(T), B, tC_i])
        B_ub = np.array([-self.LB_t, self.LB_t, -self.LB_t])
        # x?????????
        x = (0,1)
        res = op.linprog(c,A_ub,B_ub,bounds=x)
        return res.fun
# ??????????????????
def openVector(dir, sp="\t"):
    list = []
    with open(dir, encoding='utf-8') as file:
         lines = file.readlines()
         for line in lines:
             nameandvector = line.strip().split(sp)
             list.append(json.loads(nameandvector[1]))
    return list

if __name__ == '__main__':
    dirInstance = "YAGO39K\\Train\\instance2id.txt"
    instanceIdNum, instanceList_ = SetE.openDetailsAndId(dirInstance)
    dirConcept = "YAGO39K\\Train\\concept2id.txt"
    conceptIdNum, conceptList_ = SetE.openDetailsAndId(dirConcept)
    dirRelation = "YAGO39K\\Train\\relation2id.txt"
    relationIdNum, relationList_ = SetE.openDetailsAndId(dirRelation)
    # ????????????
    dirConceptVector = "YAGO39K\\Vector\\conceptVector.txt"
    conceptList = openVector(dirConceptVector)
    dirInstanceVector = "YAGO39K\\Vector\\instanceVector.txt"
    instanceList = openVector(dirInstanceVector)
    dirRelationVector = "YAGO39K\\Vector\\relationVector.txt"
    relationList = openVector(dirRelationVector)
    # ???????????????
    dirtriposTest = "YAGO39K\\Test\\triple2id_positive.txt"
    dirsubclassofposTest = "YAGO39K\\Test\\subClassOf2id_positive.txt"
    dirinstanceofposTest = "YAGO39K\\Test\\instanceOf2id_positive.txt"

    dirtrinegTest = "YAGO39K\\Test\\triple2id_negative.txt"
    dirsubclassofnegTest = "YAGO39K\\Test\\subClassOf2id_negative.txt"
    dirinstanceofnegTest = "YAGO39K\\Test\\instanceOf2id_negative.txt"
    # ???????????????
    tripleNumposTest, tripleListposTest = SetE.openTrainTri(dirtriposTest, instanceList_, relationList_)
    subclassofNumposTest, subclassofListposTest = SetE.openTrainBin(dirsubclassofposTest, 0, instanceList_, conceptList_)

    instanceofNumposTest, instanceofListposTest = SetE.openTrainBin(dirinstanceofposTest, 1, instanceList_, conceptList_)

    #???????????????
    tripleNumnegTest, tripleListnegTest = SetE.openTrainTri(dirtrinegTest, instanceList_, relationList_)
    subclassofNumnegTest, subclassofListnegTest = SetE.openTrainBin(dirsubclassofnegTest, 0, instanceList_, conceptList_)
    instanceofNumnegTest, instanceofListnegTest = SetE.openTrainBin(dirinstanceofnegTest, 1, instanceList_, conceptList_)

    print("????????????")
    test = Classification(conceptList, instanceList, relationList, instanceofListposTest, instanceofListnegTest, subclassofListposTest, subclassofListnegTest, tripleListposTest, tripleListnegTest,B_t = 1, B_r = 1)

    # accuracy1, precision1, recall1, F11 = test.Tripleclassification()
    # print('???????????????',accuracy1, precision1, recall1, F11)

    accuracy2, precision2, recall2, F12 = test.Instanceofclassification()
    print('instanceof??????', accuracy2, precision2, recall2, F12)

    # accuracy3, precision3, recall3, F13 = test.Subclassofclassification()
    # print('subclassof??????', accuracy3, precision3, recall3, F13)