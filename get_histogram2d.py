from __future__ import print_function

import matplotlib.pyplot as plt
import pandas
import pandas as pd
from numpy import *
import numpy as np
import json, re,os, sys
import argparse
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


geneID_map=None
ID_to_name_map=None
mapfile='04622_geneName_map.txt'
filename='EPCdata/adj_04622.csv'

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
    geneIDs = load_real_data_from_csv()
    index=geneID_map.get(str(geneA))#得到基因A
    print("=========================",index)
    index2 = np.where(geneIDs == index)
    index2 = index2[0]
    indexb=geneID_map.get(geneB)
    indexb2 = np.where(geneIDs == indexb)
    indexb2 = indexb2[0]
    df = pd.read_csv(filename,header='infer',index_col=0)
    rpkm=df.T
    rpkm.iloc[indexb2,index2]=0
    #m=rpkm.iloc[:,0]
    #print(m)
    if index is None:
        index = int(geneA)
        geneA_x = rpkm.iloc[:,index]#取一个基因在不同细胞中的所有的表达值
    else:
        #index=int(index)
        if type(geneIDs[0])==int:
            index=int(index)
        index2 = np.where(geneIDs==index)#只有条件 (condition)，没有x和y，则输出索引的坐标
        indexb2=np.where(geneIDs==indexb)
        index = index2[0]
        indexb=indexb2[0]
        geneA_x = rpkm.iloc[:,index]#iloc通过行号来取行数据
        #print(';;;;;;;;;;;;;;;',geneA_x)
    geneA_x=geneA_x.to_numpy().reshape(-1)#reshape给数组一个新的形状而不改变其数据；to_numpy将数据格式转换为一个NumPy数组
    #print('[[[[[[[[[[[[[[[[',geneA_x)
    geneA_x[indexb-len(geneIDs)]=0
    #print('///////////////',geneA_x)
    return geneA_x
def get_expr_by_networki_geneName1( geneA):  #for simulation, for liver#输出基因A的表达值
    geneID_map = get_gene_list()
    index=geneID_map.get(str(geneA))#得到基因A
    rpkm = pd.read_csv(filename,header='infer',index_col=0)
    rpkm=rpkm.T
    geneIDs=load_real_data_from_csv()
    #print(geneIDs)
    if index is None:
        index = int(geneA)
        geneA_x = rpkm.iloc[:,index]#取一个基因在不同细胞中的所有的表达值
    else:
        #index=int(index)
        if type(geneIDs[0])==int:
            index=int(index)
        index2 = np.where(geneIDs==index)#只有条件 (condition)，没有x和y，则输出索引的坐标
        index = index2[0]
        geneA_x = rpkm.iloc[:,index]#iloc通过行号来取行数据
    geneA_x=geneA_x.to_numpy().reshape(-1)#reshape给数组一个新的形状而不改变其数据；to_numpy将数据格式转换为一个NumPy数组
    #print('lllllllllllllllll',geneA_x)
    return geneA_x
def hisgram(x_geneA,x_geneB):
    if x_geneA is not None:
        if x_geneB is not None:
            x_tf = x_geneA
            x_gene = x_geneB

            H_T = histogram2d(x_tf, x_gene, bins=10)  # 创立直方图；将直方图设置成32*32的，求tf和gene落在每一个小格格的数量

            H = H_T[0].T
            HT = (log10(H / len(x_tf) + 10 ** -4) + 4) / 4  # 求相对于整个tf的概率
            #HT=np.append(x_tf,x_gene)
    return HT

