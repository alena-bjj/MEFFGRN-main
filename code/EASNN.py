from __future__ import print_function



import matplotlib.pyplot as plt
import pandas
import pandas as pd
from numpy import *
import numpy as np
import torch
import json, re,os, sys
import argparse
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


geneID_map=None
ID_to_name_map=None
mapfile='04622_geneName_map.txt'
filename='04622adj_matrix.csv'

def get_gene_list( ):  # 取得基因map文件
    import re
    h = {}  # h [gene symbol] = gene ID
    h2 = {}  # h2 [gene ID] = gene symbol
    s = open(mapfile, 'r')  # gene symbol ID list of sc RNA-seq
    for line in s:
        search_result = re.search(r'^([^\s]+)\s+([^\s]+)', line)

        # print("-",search_result.group(1),"-",search_result.group(2),"-")
        h[str(search_result.group(1).lower())] = str(search_result.group(2).lower())  # h [gene symbol] = gene ID;group（0）就是匹配正则表达式整体结果
        # group(1) 列出第一个括号匹配部分，group(2) 列出第二个括号匹配部分，group(3) 列出第三个括号匹配部分。
        h2[str(search_result.group(2).lower())] = str(search_result.group(1).lower())  # h2 [geneID] = gene symbol
    geneID_map = h
    s.close()
    return  geneID_map


def load_real_data_from_csv():  # 从csv文件中读取基因名
    df = pd.read_csv(filename,header='infer',index_col=0)  # 读取csv文件
    geneIDs = df.index  # 文件的列为geneID,header='infer',
    geneIDs = np.asarray(geneIDs, dtype=str)  # 更改值为str型
    for i in range(0, len(geneIDs)):
        geneIDs[i] = geneIDs[i].lower()  # 将基因名改为小写
    return geneIDs
def get_index_by_networki_geneName(geneA):  #for simulation, for liver；输出基因
    geneID_map=get_gene_list()
    index=geneID_map.get(str(geneA))

    geneIDs=load_real_data_from_csv()
    if index is None:#判断index是否被声明和定义
        index = int(geneA)
            #print("gene", geneA, "not found")
            #return None
    else:
        if type(geneIDs[0])==int:
            index=int(index)
        index2 = np.where(geneIDs==index)#np.where:只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标
        index = index2[0]
        #print('===========',index)
    return index
def get_expr_by_networki_geneName( geneA,geneB):  #for simulation, for liver#输出基因A的表达值
    geneID_map = get_gene_list()
    index=geneID_map.get(str(geneA))#得到基因A
    indexb=geneID_map.get(geneB)
    df = pd.read_csv(filename,header='infer',index_col=0)
    rpkm=df.T
    geneIDs=load_real_data_from_csv()
    if index is None:
        index = int(geneA)
        geneA_x = rpkm.iloc[:,index]#取一个基因在不同细胞中的所有的表达值
        geneA_x1 = rpkm.iloc[ index,:]  # 取一个基因在不同细胞中的所有的表达值
    else:
        #index=int(index)
        if type(geneIDs[0])==int:
            index=int(index)
        index2 = np.where(geneIDs==index)#只有条件 (condition)，没有x和y，则输出索引的坐标
        indexb2=np.where(geneIDs==indexb)
        index = index2[0]
        indexb=indexb2[0]
        geneA_x = rpkm.iloc[:,index]#iloc通过行号来取行数据
        geneA_x1= rpkm.iloc[index,:]  # iloc通过行号来取行数据
        #print(';;;;;;;;;;;;;;;',geneA_x)
    geneA_x=geneA_x.to_numpy().reshape(-1)#reshape给数组一个新的形状而不改变其数据；to_numpy将数据格式转换为一个NumPy数组
    geneA_x1 = geneA_x1.to_numpy().reshape(-1)
    return geneA_x,geneA_x1


def get_Regularition(geneA,geneB):
    geneID_map = get_gene_list()
    geneIDs = load_real_data_from_csv()
    indexA = geneID_map.get(str(geneA))  # 得到基因A
    index1 = np.where(geneIDs == indexA)
    indexa=index1[0]
    indexB = geneID_map.get(str(geneB))
    index2 = np.where(geneIDs == indexB)
    indexb = index2[0]
    df = pd.read_csv(filename, header='infer', index_col=0)
    rpkm = df.T
    rpkm.iloc[indexb,indexa]=0
    countm = []
    countn = []
    for i in range(len(geneIDs)):
        M = np.zeros((len(geneIDs)), dtype=int)  # 基因i的奖励矩阵的值
        N = np.zeros((len(geneIDs)), dtype=int)  # 转置基因i的奖励矩阵的值
        #求基因m1以及m1的转置n1与其他基因的建立矩阵的值
        m1 = rpkm.iloc[:, i]
        m1 = m1.to_numpy().reshape(-1)
        n1 = rpkm.iloc[i, :]
        n1 = n1.to_numpy().reshape(-1)
        for j in range(len(geneIDs)):
            m2 = rpkm.iloc[:, j]
            m2 = m2.to_numpy().reshape(-1)
            # print(a1)
            if (m1[j] == 1):
                M[j] = M[j] + 9
            if (m2[i] == 1):
                M[j] = M[j] + 9
            for j1 in range(len(geneIDs)):
                if (m1[j1] == 0 and m2[j1] == 0 and j1 != j and j1 != i) or (
                        m1[j1] == 1 and m2[j1] == 1 and j1 != j and j1 != i):
                    M[j] = M[j] + 1

                if (m1[j1] == 0 and (m2[j1] == 1 and j1 != i)) or ((m1[j1] == 1 and j1 != j) and m2[j1] == 0):
                    M[j] = M[j] - 1
        for j in range(len(geneIDs)):
            n2 = rpkm.iloc[j, :]
            n2 = n2.to_numpy().reshape(-1)
            if (n1[j] == 1):
                N[j] = N[j] + 9
            if (n2[i] == 1):
                N[j] = N[j] + 9
            for j1 in range(len(geneIDs)):
                if (n1[j1] == 0 and n2[j1] == 0 and j1 != j and j1 != i) or (
                        n1[j1] == 1 and n2[j1] == 1 and j1 != j and j1 != i):
                    N[j] = N[j] + 1

                if (n1[j1] == 0 and (n2[j1] == 1 and j1 != i)) or ((n1[j1] == 1 and j1 != j) and n2[j1] == 0):
                    N[j] = N[j] - 1

        M,N=M*2,N*2
        M[i]=0
        N[i]=0

        countm.append(M)
        countn.append(N)




    finalMM=[]
    finalNN=[]
    for i in range(len(geneIDs)):
        MM = np.zeros((len(geneIDs)), dtype=float)  # 正则化后的基因a的建立矩阵的值
        NN = np.zeros((len(geneIDs)), dtype=float)  # 正则化后的转置基因a的建立矩阵的值
        for j in range(len(geneIDs)):
            norm = (max(countm[i]) - min(countm[i])) * (max(countm[j]) - min(countm[j]))
            MM[j] = ((countm[i][j] - min(countm[i])) * (countm[i][j] - min(countm[j]))) / norm
            MM[j] = MM[j] * MM[j]
        MM[i] = 1
        finalMM.append(MM)

        for j in range(len(geneIDs)):
            norm1 = (max(countn[i]) - min(countn[i])) * (max(countn[j]) - min(countn[j]))
            NN[j] = ((countn[i][j] - min(countn[i])) * (countn[i][j] - min(countn[j]))) / norm1
            NN[j] = NN[j] * NN[j]
        NN[i] = 1
        finalNN.append(NN)
    return rpkm,finalMM,finalNN
def SNN(geneA,geneB):
    rpkm,mm,nn=get_Regularition(geneA,geneB)
    M = []
    N=[]
    km1=[]
    kn1=[]
    for i in range(len(mm)):
        m = np.zeros((len(mm)), dtype=float)
        mm2 = np.zeros((len(mm)), dtype=float)
        mm1 = sorted(mm[i], reverse=True)#得到降序后的值
        #indexmm = mm[i].argsort()[::-1]#得到降序后的索引
        indexmm = [index for index, value in sorted(list(enumerate(mm[i])), key=lambda x: x[1], reverse=True)]
        si = floor(len(mm) / 10)
        kl = 0
        for i in range(len(mm1)):
            mm2[i] = floor(mm1[i] * 10) / 10
        mk1 = pd.value_counts(mm2).sort_index()#得到不同值的数量
        mk = np.array(mk1)
        lenmm1 = len(mk1)
        while kl < si:
            kf = kl
            kl = kl + mk[lenmm1-1]
            lenmm1 = lenmm1 - 1
        if kl < 2 * si:
            km = kl
        elif kf > si / 2:
            km = kf
        else:
            km = si
        km1.append(km)

        for j in range(0, int(km)):
            m[indexmm[j]] = mm1[j]
        M.append(m)
    for i in range(len(mm)):
        for j in range(len(mm)):
            if M[i][j]>=M[j][i]:
                M[i][j]=M[i][j]
            else :
                M[i][j] = M[j][i]
    for i in range(len(nn)):
        n = np.zeros((len(nn)), dtype=float)
        nn2 = np.zeros((len(nn)), dtype=float)
        nn1 = sorted(nn[i], reverse=True)#得到降序后的值

        #indexnn = nn[i].argsort()[::-1]#得到降序后的索引
        indexnn = [index for index, value in sorted(list(enumerate(nn[i])), key=lambda x: x[1], reverse=True)]
        sin = floor(len(nn) / 10)
        kl = 0
        for i in range(len(nn1)):
            nn2[i] = floor(nn1[i] * 10) / 10
        nk1 = pd.value_counts(nn2).sort_index()#得到不同值的数量
        nk = np.array(nk1)
        lennn1 = len(nk1)
        while kl < sin:
            kf = kl
            kl = kl + nk[lennn1-1]
            lennn1 = lennn1 - 1
        if kl < 2 * sin:
            kn = kl
        elif kf > sin / 2:
            kn = kf
        else:
            kn = sin
        kn1.append(kn)
        for j in range(0, int(kn)):
            n[indexnn[j]] = nn1[j]
        N.append(n)
    for i in range(len(nn)):
        for j in range(len(nn)):
            if N[i][j]>=N[j][i]:
                N[i][j]=N[i][j]
            else :
                N[i][j] = N[j][i]
    return rpkm,M,N,km1,kn1
def get_finalmatrix(geneA,geneB):
    rpkm,M,N,km1,kn1=SNN(geneA,geneB)
    geneID_map = get_gene_list()
    geneIDs = load_real_data_from_csv()
    indexA = geneID_map.get(str(geneA))  # 得到基因A
    index1 = np.where(geneIDs == indexA)
    indexa = index1[0]
    indexB = geneID_map.get(str(geneB))
    index2 = np.where(geneIDs == indexB)
    indexb = index2[0]
    r=0.9
    Y=[]
    Yn=[]
    for i in range(len(km1)):
        sort_m = sorted(M[i], reverse=True)  # 得到降序后的值
        #index_m = M[i].argsort()[::-1]  # 得到降序后的索引
        index_m = [index for index, value in sorted(list(enumerate(M[i])), key=lambda x: x[1], reverse=True)]
        sun_m=np.sum(sort_m[:int(km1[i])])
        w=[]
        y=np.zeros(len(km1),dtype=float)
        for j in range(int(km1[i])):
            matrixm1 = rpkm.iloc[:, index_m[j]]
            matrixm1 = matrixm1.to_numpy().reshape(-1)
            a=r**j
            w1=a*sort_m[j]
            y=y+w1*matrixm1
           # print(y)
            w.append(w1)
        y=y/sun_m
        Y.append(y)
    for i1 in range(len(kn1)):
        sort_n = sorted(N[i1], reverse=True)  # 得到降序后的值
        #index_n = (-N[i1]).argsort()# 得到降序后的索引
        index_n = [index for index, value in sorted(list(enumerate(N[i1])), key=lambda x: x[1], reverse=True)]
        sun_n = np.sum(sort_n[:int(kn1[i1])])
        wn = []
        yn = np.zeros(len(kn1), dtype=float)
        for j1 in range(int(kn1[i1])):
            matrixn1 = rpkm.iloc[index_n[j1],:]
            matrixn1 = matrixn1.to_numpy().reshape(-1)
            an = r ** j1
            wn1 = an * sort_n[j1]
            yn = yn + wn1 * matrixn1
            wn.append(wn1)
        yn = yn / sun_n
        Yn.append(yn)
    matrix=[]
    for c in range(len(kn1)):
        hang = np.zeros(len(km1), dtype=float)
        for d in range(len(kn1)):
            hang1 = np.zeros(len(km1), dtype=float)
            m=Y[c][d]*0.5+Yn[d][c]*0.5
            hang1[d]=m
            hang=hang+hang1
        matrix.append(hang)
    ha = np.zeros(len(km1), dtype=float)
    hb = np.zeros(len(km1), dtype=float)
    finalmatrix1 = rpkm.iloc[:, int(indexa)]
    finalmatrix1 = finalmatrix1.to_numpy().reshape(-1)
    finalmatrix2 = rpkm.iloc[:, int(indexb)]
    finalmatrix2 = finalmatrix2.to_numpy().reshape(-1)
    for bb in range(len(km1)):
        if matrix[int(indexa)][bb] > finalmatrix1[bb]:
            ha[bb] = matrix[int(indexa)][bb]
        else:
         ha[bb] = finalmatrix1[bb]
    for aa in range(len(km1)):
        if matrix[int(indexb)][aa] > finalmatrix2[aa]:
            hb[aa] = matrix[int(indexb)][aa]
        else:
         hb[aa] = finalmatrix2[aa]
   # ha[indexb]=0

    # print(hb)
    return  ha,hb

def hisgram(x_geneA,x_geneB):
    if x_geneA is not None:
        if x_geneB is not None:
            x_tf = x_geneA
            x_gene = x_geneB

            H_T = histogram2d(x_tf, x_gene, bins=10)  # 创立直方图；将直方图设置成32*32的，求tf和gene落在每一个小格格的数量

            H = H_T[0].T
            HT = (log10(H / len(x_tf) + 10 ** -4) + 4) / 4  # 求相对于整个tf的概率

    return HT









