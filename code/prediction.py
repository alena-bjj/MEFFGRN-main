from __future__ import print_function

import random

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import os
import MEFFGRNmodel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef, auc
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint  # 回调函数
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve
from datasetSplit import c1_index,c2_index,c3_index,c4_index,c5_index,index
from sklearn.model_selection import train_test_split,KFold
import warnings
import csv

warnings.filterwarnings("ignore")

dir='D:/EPCGRN/'
data_path = dir+'04622_representation/version11'
data_path1=dir+'04622-adj-zq_representation/version11'

matrix_data = np.load(data_path + '/0_xdata.npy')
matrix_data1=np.load(data_path1+'/0_xdata.npy')
label_data = np.load(data_path + '/0_ydata.npy')
pairs_data=np.load(data_path+'/0_zdata.npy')
y_test=label_data
z_test=pairs_data
num_pairs = len(label_data)

#x_train=matrix_data[index]
x_train=matrix_data[index]
x_train1=matrix_data1[index]
y_train=label_data[index]
x_train2 = list(zip(x_train, x_train1))
(trainXX1, testXX1, trainYY, testYY) = train_test_split(x_train2, y_train, test_size=0.2, random_state=1,
                                                              shuffle=True)

trainXX = []
trainadj = []
testXX = []
testadj = []
for i in range(len(trainXX1)):
    trainXX.append(trainXX1[i][0])
    trainadj.append(trainXX1[i][1])
trainXX = np.array(trainXX)
trainadj = np.array(trainadj)
for j in range(len(testXX1)):
    testXX.append(testXX1[j][0])
    testadj.append(testXX1[j][1])
testXX = np.array(testXX)
testadj = np.array(testadj)
x_train_list = []
model = MEFFGRNmodel.construct_model(trainXX,trainadj)


predict_output_dir='Predict/'

if not os.path.isdir(predict_output_dir):
    os.makedirs(predict_output_dir)
x_test_list=[]
x_test=matrix_data
x_test1=matrix_data1
n = x_test.shape[1]
for j in range(0,n):
  x_test_list.append(x_test[:,j,:,:,np.newaxis])

x_test_list1 = []
m = x_test.shape[1]
for i in range(0, m):
    x_test_list1.append(x_test1[:, i, :, :, np.newaxis])

model_path='train-model/keras_MEFFGRN.h5'
model.load_weights(model_path)
print('load model and predict')
y_predict = model.predict([x_test_list, x_test_list1])
np.save(predict_output_dir + 'end_y_predict.npy', y_predict)
np.savetxt(predict_output_dir+"end_y_predict.csv", y_predict, delimiter=",")

np.save(predict_output_dir + 'end_y_test.npy', y_test)
np.savetxt(predict_output_dir+ 'end_y_test.csv', y_test, delimiter=",")
print(z_test)
df = pd.DataFrame(z_test)
df.to_csv(predict_output_dir + 'end_z_test.csv')