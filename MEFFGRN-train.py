from __future__ import print_function

import random

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import os

import MEFFGRNmodel
import numpy as np
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

def load_index(file):
    with open(file,'r') as f:
        csv_r=list(csv.reader(f,delimiter='\n'))
    return np.array(csv_r).flatten().astype(int)

def auroc(y_true, y_pred):
    return tf.py_func(metrics.roc_auc_score, (y_true, y_pred), tf.double)

def aupr_cal(y_true, y_pred):
    precision, recall, thresholds_PR = metrics.precision_recall_curve(y_true, y_pred)
    return metrics.auc(recall, precision)

def aupr(y_true, y_pred):
    # precision, recall, thresholds_PR = metrics.precision_recall_curve(y_true, y_pred)
    # AUPR = metrics.auc(recall, precision)
    return tf.py_func(aupr_cal, (y_true, y_pred), tf.double)

def scores(y_test,y_pred,th=0.5):
    y_predlabel = [(0 if item<th else 1) for item in y_pred]
    tn,fp,fn,tp = confusion_matrix(y_test,y_predlabel).flatten()
    SPE = tn*1./(tn+fp)
    MCC = matthews_corrcoef(y_test,y_predlabel)
    Recall = recall_score(y_test, y_predlabel)
    Precision = precision_score(y_test, y_predlabel)
    F1 = f1_score(y_test, y_predlabel)
    Acc = accuracy_score(y_test, y_predlabel)
    AUC = roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    AUPR = auc(recall_aupr, precision_aupr)
    return [Recall,SPE,Precision,F1,MCC,Acc,AUC,AUPR]

####################################### parameter settings
batch_size = 2 # mini batch for training
num_classes = 2  # categories of labels
epochs = 200


# data_path = 'E:/Papers/20210124-DGRNS/500genes/mHSC-E-Nonspecific-LogPearson/WindowSize131-TF10-Target10-Lag32/FullMatrix_TF'
# model_save_dir = 'E:/Papers/20210124-DGRNS/500genes/mHSC-E-Nonspecific-LogPearson/WindowSize131-TF10-Target10-Lag32/DGRNS-model-Independent/'


#os.system('hebing04622.py')
dir='D:/GRN/'
data_path = dir+'04622_representation/version11'
data_path1= dir+'04622-adj-zq_representation/version11'
model_save_dir =dir+'train-model'
if not os.path.isdir(model_save_dir):
    os.makedirs(model_save_dir)

matrix_data1 = np.load(data_path + '/0_xdata.npy')
matrix_data2 = np.load(data_path1 + '/0_xdata.npy')
#matrix_data=np.concatenate((matrix_data1,matrix_data2),axis=3)
label_data = np.load(data_path + '/0_ydata.npy')
pairs_data=np.load(data_path+'/0_zdata.npy')
num_pairs = len(label_data)

# train_index=load_index(data_path+'/train_index.txt')
# val_index=load_index(data_path+'/val_index.txt')
# test_index=load_index(data_path+'/test_index.txt')




# split training and validation systematically
# train_val_index=np.append(train_index,val_index)
# train_val_data=matrix_data[train_val_index]
# train_val_label=label_data[train_val_index]
# x_train, x_val,y_train, y_val = train_test_split(train_val_data, train_val_label, test_size=0.2, random_state=1)



'''print(x_train.shape, 'x_train samples')
print(x_test.shape, 'x_test samples')
print(y_train.shape, 'y_train samples')
print(x_train.shape[1:])
print(y_test.shape, 'y_test samples')'''

#calculate running time




all_acc = []
all_val_acc = []
all_loss = []
all_val_loss = []

for ki in range (30):
    import os
    print("\n第{}次训练..........\n".format(ki + 1))
    os.system('datasetSplit.py')
    model_name = 'keras_DeepDRIM{}.h5'.format(ki)
    kf = KFold(n_splits=5, shuffle=True)
    AUROCs = []
    AUPRs = []
    Recalls = []
    Precisions = []
    F1s = []
    Accs = []
    x_train=matrix_data1[index]
    x_train1=matrix_data2[index]
    y_train=label_data[index]
    x_train2 = list(zip(x_train, x_train1))
    (trainXX1, testXX1, trainYY, testYY) = train_test_split(x_train2, y_train, test_size=0.2, random_state=1,
                                                              shuffle=True)

    #, stratify=y_train

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
    n = trainXX.shape[1]
    print(n)
    for j in range(0, n):
        x_train_list.append(trainXX[:, j, :, :, np.newaxis])



    x_val_list = []
    l = testXX.shape[1]
    for i in range(0, l):
        x_val_list.append(testXX[:, i, :, :, np.newaxis])

    x_train_list1 = []
    j = trainXX.shape[1]
    for j in range(0, n):
        # x_train_list.append(trainXX[:, j, :, :, np.newaxis])
        x_train_list1.append(trainadj[:, j, :, :, np.newaxis])

    x_val_list1 = []
    l = testXX.shape[1]
    for i in range(0, l):
        x_val_list1.append(testadj[:, i, :, :, np.newaxis])
        # x_val_list.append(testXX[:, i, :, :, np.newaxis])


    model = MEFFGRNmodel.construct_model(trainXX,trainadj)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, verbose=0.0001, mode='auto')
    checkpoint1 = ModelCheckpoint(filepath=model_save_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss',
                                        verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    checkpoint2 = ModelCheckpoint(filepath=model_save_dir + '/weights{}.hdf5'.format(ki), monitor='val_accuracy', verbose=1,
                                        save_best_only=True, mode='auto', period=1)
    patience=10
    reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.001, patience=int(patience / 2), verbose=1)  # 性能没有提升时，调整学习率
    callbacks_list = [checkpoint2, early_stopping,reduce_lr]
    history = model.fit([x_train_list,x_train_list1], trainYY, validation_data=([x_val_list,x_val_list1] , testYY), batch_size=batch_size,
                            epochs=epochs,  # validation_split=0.2,
                            shuffle=True, callbacks=callbacks_list)  # validation_split用来指定训练集的一定比例数据作为验证集
    model_path = os.path.join(model_save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    all_acc.append(acc)
    all_val_acc.append(val_acc)
    all_loss.append(loss)
    all_val_loss.append(val_loss)

epochs, avg_acc, avg_val_acc, avg_loss, avg_val_loss = DeepDRIMmodel.average_acc_loss(all_acc,
                                                                                      all_val_acc, all_loss, all_val_loss)
print('平均结果计算完成')

# 绘制 acc loss曲线(平滑)
acc_name = 'ko04620  accuracy_smoothed' +  '.png'
loss_name = 'ko04620 loss_smoothed' +  '.png'
acc_title = ' ko04620 NET_smoothed acc'
loss_title = ' ko04620 NET_smoothed loss'
plt.figure()
plt.plot(epochs, DeepDRIMmodel.smooth_curve(avg_acc), label='Training  accuracy')
plt.plot(epochs, DeepDRIMmodel.smooth_curve(avg_val_acc), label='Validation  accuracy')
plt.title(acc_title)
plt.legend(loc='lower right')
plt.savefig(acc_name, dpi=600)
plt.figure()

plt.plot(epochs, DeepDRIMmodel.smooth_curve(avg_loss), label='Training  loss')
plt.plot(epochs, DeepDRIMmodel.smooth_curve(avg_val_loss), label='Validation  loss')
plt.title(loss_title)
plt.legend(loc='lower left')
plt.savefig(loss_name, dpi=600)
# plt.show()

acc_name = 'ko04620  accuracy' + '.png'
loss_name = 'ko04620  loss' + '.png'
acc_title = 'ko04620 NET_smoothed acc '
loss_title = 'ko04620 NET_smoothed loss '
plt.figure()
plt.plot(epochs, avg_acc, label='Training  accuracy')
plt.plot(epochs, avg_val_acc, label='Validation  accuracy')
plt.title(acc_title)
plt.legend(loc='lower right')
plt.savefig(acc_name, dpi=600)
plt.figure()

plt.plot(epochs, DeepDRIMmodel.smooth_curve(avg_loss), label='Training  loss')
plt.plot(epochs, DeepDRIMmodel.smooth_curve(avg_val_loss), label='Validation  loss')
plt.title(loss_title)
plt.legend(loc='lower left')
plt.savefig(loss_name, dpi=600)







