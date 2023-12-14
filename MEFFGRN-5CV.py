from __future__ import print_function

import random

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import tensorflow as tf
import os
import EASNN
import MEFFGRNmodel
import get_histogram2d


import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef, auc
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve
from datasetSplit import c1_index,c2_index,c3_index,c4_index,c5_index
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
    #AUC = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    #precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    #AUPR = auc(recall_aupr, precision_aupr)
    precision, recall, thresholds_PR = metrics.precision_recall_curve(y_test, y_pred)
    AUPR = metrics.auc(recall, precision)
    return [Recall,SPE,Precision,F1,MCC,Acc,AUC,AUPR]


####################################### parameter settings
batch_size = 4  # mini batch for training
num_classes = 2  # categories of labels
epochs = 200

model_name = 'keras_DeepDRIM.h5'
# data_path = 'E:/Papers/20210124-DGRNS/500genes/mHSC-E-Nonspecific-LogPearson/WindowSize131-TF10-Target10-Lag32/FullMatrix_TF'
# model_save_dir = 'E:/Papers/20210124-DGRNS/500genes/mHSC-E-Nonspecific-LogPearson/WindowSize131-TF10-Target10-Lag32/DGRNS-model-Independent/'
dir='D:/MEFFGRN/'
data_path = dir+'04622_representation/version11'
#data_path1 = dir+'04622-adj_representation/version11'
data_path2=dir+'04622-adj-zq_representation/version11'
model_save_dir =dir+'train-model'
if not os.path.isdir(model_save_dir):
    os.makedirs(model_save_dir)
#matrix_data = np.load(data_path + '/xdata.npy')
matrix_data1 = np.load(data_path + '/0_xdata.npy')
matrix_data2=np.load(data_path2+'/0_xdata.npy')
#matrix_data=list(zip(matrix_data1,matrix_data4))
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





netavgAUROCs = []  # 存放一个网络10次5CV的平均AUC
netavgAUPRs = []
netavgRecalls = []
netavgSPEs = []
netavgPrecisions = []
netavgF1s = []
netavgMCCs = []
netavgAccs = []

for ki in range (10):
    import os
    print("\n第{}次5折交叉验证..........\n".format(ki + 1))
    os.system('datasetSplit.py')
    kf = KFold(n_splits=5, shuffle=True)
    #model_name = 'keras_DeepDRIM{}.h5'.format(ki)
    AUROCs = []
    AUPRs = []
    Recalls = []
    Precisions = []
    F1s = []
    Accs = []
    index = [c1_index, c2_index, c3_index, c4_index, c5_index]
    for i in range(0, 5):  # 调用split方法切分数据
        # 6.1 划分4:1训练集 测试集
        #             print('train_index:%s , test_index: %s ' %(train_index,test_index))

        model_name = 'keras_MEFFGRN{}'.format(ki)+'-{}.h5'.format(i)
        test_index = index[i]
        random.shuffle(test_index)
        train_index = []
        for a in range(0, 5):
            if a != i:
                train_index = train_index + index[a]
        random.shuffle(train_index)
        x_train, x_test = matrix_data1[train_index], matrix_data1[test_index]  # testX.shape (71, 1, 620, 1)
        x_train1,x_test1=matrix_data2[train_index],matrix_data2[test_index]
        y_train, y_test = label_data[train_index], label_data[test_index]
        z_train, z_test = pairs_data[train_index], pairs_data[test_index]

        for i in range(len(x_test)):
            m=y_test[i]
            if m==1:
                n=z_test[i]
                a, b = str(n).split(',')
                #x_geneA = get_histogram2d.get_expr_by_networki_geneName(a, b)
                #x_geneB = get_histogram2d.get_expr_by_networki_geneName1(b)
                x_geneA, x_geneB=EASNN.get_finalmatrix(a,b)
                HT = EASNN.hisgram(x_geneA, x_geneB)
                x_test1[i][0]=HT
        print(y_test)
        print(z_test)

        a = 0
        for i in range(len(y_test)):
            if y_test[i] == 1:
                a = a + 1
        print(a)
        n = x_train.shape[1]
        x_train2=list(zip(x_train,x_train1))

        (trainXX1, testXX1, trainYY, testYY) = train_test_split(x_train2, y_train, test_size=0.2, random_state=1,
                                                              shuffle=True)
        #, stratify=y_train
        trainXX=[]
        trainadj=[]
        testXX=[]
        testadj=[]
        #x_test=[]
        #valadj=[]
        for i in range(len(trainXX1)):
            trainXX.append(trainXX1[i][0])
            trainadj.append(trainXX1[i][1])
        trainXX=np.array(trainXX)
        trainadj=np.array(trainadj)
        for j in range(len(testXX1)):
            testXX.append(testXX1[j][0])
            testadj.append(testXX1[j][1])
        testXX=np.array(testXX)
        testadj=np.array(testadj)
        x_train_list = []
        n = trainXX.shape[1]
        for j in range(0, n):
            #x_train_list.append(trainXX[:, j, :, :, np.newaxis])
            x_train_list.append(trainXX[:, j, :, :, np.newaxis])

        x_test_list = []
        m = x_test.shape[1]
        for i in range(0, m):
            x_test_list.append(x_test[:, i, :, :, np.newaxis])
            #x_test_list.append(x_test[:, i, :, :, np.newaxis])


        x_val_list = []
        l = testXX.shape[1]
        for i in range(0, l):
            x_val_list.append(testXX[:, i, :, :, np.newaxis])
            #x_val_list.append(testXX[:, i, :, :, np.newaxis])

        x_train_list1 = []
        n = trainXX.shape[1]
        for j in range(0, n):
            # x_train_list.append(trainXX[:, j, :, :, np.newaxis])
            x_train_list1.append(trainadj[:, j, :, :, np.newaxis])

        x_test_list1 = []
        m = x_test.shape[1]
        for i in range(0, m):
            x_test_list1.append(x_test1[:, i, :, :, np.newaxis])
            # x_test_list.append(x_test[:, i, :, :, np.newaxis])

        x_val_list1 = []
        l = testXX.shape[1]
        for i in range(0, l):
            x_val_list1.append(testadj[:, i, :, :, np.newaxis])
            # x_val_list.append(testXX[:, i, :, :, np.newaxis])
        model = MEFFGRNmodel.construct_model(trainXX,trainadj)
        #adamw = keras.optimizers.Adam(lr=4e-3, beta_1=0.9, beta_2=0.999, decay=0.05)DeepDRIMmodelduomotaiIMFDNN.get_loss(score_1,y_test)
        score_1 = model.predict([x_test_list, x_test_list1])
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        checkpoint1 = ModelCheckpoint(filepath=dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss',
                                        verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        checkpoint2 = ModelCheckpoint(filepath=dir + '/weights1--.hdf5', monitor='val_accuracy', verbose=1,
                                        save_best_only=True, mode='auto', period=1)
        patience = 10
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.001, patience=int(patience / 2), verbose=1)  # 性能没有提升时，调整学习率
        callbacks_list = [checkpoint2, early_stopping,reduce_lr]
        model_path = os.path.join(model_save_dir, model_name)
        model.save(model_path)
        y_predict = model.predict([x_test_list,x_test_list1])
        history = model.fit([x_train_list,x_train_list1], trainYY, validation_data=([x_val_list,x_val_list1] ,testYY), batch_size=batch_size,
                            epochs=epochs,  # validation_split=0.2,
                            shuffle=True, callbacks=callbacks_list)  # validation_split用来指定训练集的一定比例数据作为验证集
        score_1 = model.predict([x_test_list,x_test_list1])

        Recall,SPE,  Precision, F1,MCC,Acc ,  aucROC, AUPR = scores(y_test, score_1,th=0.5)
        AUROCs.append(aucROC)
        AUPRs.append(AUPR)
        Recalls.append(Recall)
        Precisions.append(Precision)
        F1s.append(F1)
        Accs.append(Acc)
        print('\nAUROCs:')
        print(AUROCs)
        print('\n')
        # 一次五折交叉验证（1个网络5折的AUC）的平均AUC值
        avg_AUROC = np.mean(AUROCs)
        avg_AUPR = np.mean(AUPRs)
        avg_Recalls = np.mean(Recalls)
        avg_Precisions = np.mean(Precisions)
        avg_F1s = np.mean(F1s)
        avg_Accs = np.mean(Accs)
    avg_AUROC = np.mean(AUROCs)
    avg_AUPR = np.mean(AUPRs)
    avg_Recalls = np.mean(Recalls)
    avg_Precisions = np.mean(Precisions)
    avg_F1s = np.mean(F1s)
    avg_Accs = np.mean(Accs)
    print('5CV平均值',avg_AUROC)

    # 10次5CV的AUC值，有10个值
    netavgAUROCs.append(avg_AUROC)  # 10个AUC值，长度为10
    netavgAUPRs.append(avg_AUPR)
    netavgRecalls.append(avg_Recalls)
    netavgPrecisions.append(avg_Precisions)
    netavgF1s.append(avg_F1s)
    netavgAccs.append(avg_Accs)
print('十次五折交叉验证的所有AUC值--------------------------------------------')
print(netavgAUROCs)
print ( '---------------------------------------------------------------------')
print('十次五折交叉验证的所有AUPR值--------------------------------------------')
print(netavgAUPRs)
print ( '---------------------------------------------------------------------')
    # 10次5CV的AUC平均值、标准差，有1个值
AUROC_mean = np.mean(netavgAUROCs)
AUROC_std = np.std(netavgAUROCs, ddof=1)
AUPR_mean = np.mean(netavgAUPRs)
AUPR_std = np.std(netavgAUPRs)
Recall_mean = np.mean(netavgRecalls)
Recall_std = np.std(netavgRecalls)
Precision_mean = np.mean(netavgPrecisions)
Precision_std = np.std(netavgPrecisions)
F1_mean = np.mean(netavgF1s)
F1_std = np.std(netavgF1s)
Acc_mean = np.mean(netavgAccs)
Acc_std = np.std(netavgAccs)
print('10次平均AUC值',AUROC_mean)
print('AUC值方差',AUROC_std)
print('10次平均Aupr值',AUPR_mean)
print('AUPR值方差',AUPR_std)
print('10次平均ACC值',Acc_mean)
print('Acc值方差',Acc_std)
print('10次平均Precision值',Precision_mean)
print('Precision值方差',Precision_std)
print('10次平均F1值',F1_mean)
print('F1值方差',F1_std)
print('10次Recall值',Recall_mean)
print('Recall值方差',Recall_std)






