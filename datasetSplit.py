import numpy as np
import random
import csv

def split_index(all_index):
    random.shuffle(all_index)
    part = len(all_index) // 5
    t1_index = all_index[:1 * part]
    t2_index = all_index[1 * part:2 * part]
    t3_index = all_index[2 * part:3 * part]
    t4_index = all_index[3 * part:4 * part]
    t5_index = all_index[4* part:]

    return t1_index, t2_index, t3_index,t4_index,t5_index

# Split all dataset into train:validation:test = 3:1:1t4_index,t5_index
# Strategy: ratio consistent, random


data_path = 'D:/MEFFGRN/04622_representation/version11'

matrix_data = np.load(data_path + '/0_xdata.npy')
label_data = np.load(data_path + '/0_ydata.npy')
gene_pair = np.load(data_path + '/0_zdata.npy')

num_pairs = len(label_data)
pos_index=[index for index,value in enumerate(label_data) if value==1]

num=len(pos_index)
neg_index=[index for index,value in enumerate(label_data) if value==0]
#print(neg_index)
random.shuffle(neg_index)
neg_index=neg_index[0:num]




t1_index,t2_index,t3_index,t4_index,t5_index=split_index(pos_index)
t1n_index,t2n_index,t3n_index,t4n_index,t5n_index=split_index(neg_index)

c1_index=t1_index+t1n_index
c2_index=t2_index+t2n_index
c3_index=t3_index+t3n_index
c4_index=t4_index+t4n_index
c5_index=t5_index+t5n_index


index=pos_index+neg_index
random.shuffle(index)
print(index)



