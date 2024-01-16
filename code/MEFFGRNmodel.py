from __future__ import print_function

import keras
from keras.layers import Dense
from keras import  regularizers
from tensorflow.keras.optimizers import SGD
import sys
import tensorflow as tf
import numpy as np

num_classes=2

def get_single_image_model( train,train1):  # 单一图片的模型，即将单一图片通过CNN进行训练
    ############
    print("x_train.shape in single image", train.shape)
    print("x_train1.shape in single image", train1.shape)
    input_img1 = keras.layers.Input(shape=train.shape[1:])
    input_img2 = keras.layers.Input(shape=train1.shape[1:])
    x = keras.layers.Conv2D(8, (2, 2), padding='same', activation='relu')(input_img1)
    x = keras.layers.MaxPooling2D(pool_size=(1, 1), padding='same')(x)
    x = keras.layers.Dropout(0.35)(x)
    x2 = keras.layers.Flatten()(x)
    x2 = keras.layers.Dense(32)(x2)

    x1 = keras.layers.Conv2D(8, (2, 2), padding='same', activation='relu')(input_img2)
    x1 = keras.layers.MaxPooling2D(pool_size=(1, 1), padding='same')(x1)
    x1 = keras.layers.Dropout(0.35)(x1)
    x3 = keras.layers.Flatten()(x1)
    x3 = keras.layers.Dense(32)(x3)
    #print('[[[[[[[[[[[[[[[[[[[', x3[1].shape)
    y4 = x * x1
    y3 = keras.layers.Flatten()(y4)
    y = keras.layers.Conv2D(16, (2, 2), padding='same', activation='relu')(y4)
    y = keras.layers.MaxPooling2D(pool_size=(1, 1), padding='same')(y)



    y = keras.layers.Flatten()(y)
    y = keras.layers.concatenate([y, y3], axis=-1)
    y = keras.layers.Dropout(0.5)(y)
    model_out = keras.layers.Dense(32)(y)

    model_out = keras.layers.concatenate([x2, x3, model_out], axis=-1)
    #model_out = keras.layers.Dense(64)(model_out)

    return keras.Model([input_img1,input_img2], model_out)


def get_pair_image_model( train,train1):
    ############

    print("x_train.shape in single image", train.shape)
    print("x_train1.shape in single image", train1.shape)
    input_img1 = keras.layers.Input(shape=train.shape[1:])
    input_img2 = keras.layers.Input(shape=train1.shape[1:])
    x = keras.layers.Conv2D(8, (2, 2), padding='same', activation='relu')(input_img1)
    x = keras.layers.MaxPooling2D(pool_size=(1, 1), padding='same')(x)
    x = keras.layers.Dropout(0.35)(x)
    x2 = keras.layers.Flatten()(x)
    x2=keras.layers.Dense(32)(x2)

    x1 = keras.layers.Conv2D(8, (2, 2), padding='same', activation='relu')(input_img2)
    x1 = keras.layers.MaxPooling2D(pool_size=(1, 1), padding='same')(x1)
    x1 = keras.layers.Dropout(0.35)(x1)
    x3 = keras.layers.Flatten()(x1)
    x3=keras.layers.Dense(32)(x3)
    y4 = x * x1
    y3 = keras.layers.Flatten()(y4)
    y = keras.layers.Conv2D(16, (2, 2), padding='same', activation='relu')(y4)
    y = keras.layers.MaxPooling2D(pool_size=(1, 1), padding='same')(y)



    y = keras.layers.Flatten()(y)
    y = keras.layers.concatenate([y, y3], axis=-1)
    y = keras.layers.Dropout(0.5)(y)
    model_out = keras.layers.Dense(32)(y)

    model_out = keras.layers.concatenate([x2, x3, model_out], axis=-1)
    #model_out = keras.layers.Dense(64)(model_out)



    return keras.Model([input_img1, input_img2], model_out)







def construct_model( train,train2):
    print("x shape", train.shape)

    n = train.shape[1]
    x1 = train[:, 0, :, :, np.newaxis]  # 主图像

    x2 = train[:, 1:n, :, :, np.newaxis]  # 邻域图像
    x2_1 = x2[:, 0, :, :, :]

    m = train2.shape[1]
    y1 = train2[:, 0, :, :, np.newaxis]  # 主图像

    y2 = train2[:, 1:m, :, :, np.newaxis]  # 邻域图像
    y2_1 = y2[:, 0, :, :, :]

    single_image_model = get_single_image_model(x1,y1)  # 主图像通过单一模型训练
    #adj_single_image_model=get_adj_single_image_model(y1)

    print(y1.shape[1:])
    input_img_single = keras.layers.Input(shape=x1.shape[1:])
    input_adj_single=keras.layers.Input(shape=y1.shape[1:])
    single_image_out = single_image_model([input_img_single,input_adj_single])
    #adj_single_out=adj_single_image_model(input_adj_single)
    #single_image_out=keras.layers.concatenate([single_image_out1,adj_single_out],axis=-1)
    #single_image_out=LWFkeras.LWF(single_image_out1,adj_single_out)
    pair_image_model = get_pair_image_model(x2_1,y2_1)
    #adj_pair_image_model=get_adj_pair_image_model(y2_1)

    input_img = keras.layers.Input(shape=x2.shape[1:])
    pair_image_out_list = []
    input_img_whole_list = []
    pair_adj_out_list=[]
    input_adj_whole_list=[]
    input_img_whole_list.append(input_img_single)
    input_adj_whole_list.append(input_adj_single)
    input_img_multi_list = []
    input_adj_multi_list=[]
    for i in range(0, n - 1):
        input_img_multi = keras.layers.Input(shape=x2_1.shape[1:])  # 网络的输入层
        input_adj_multi=keras.layers.Input(shape=y2_1.shape[1:])
        input_img_multi_list.append(input_img_multi)
        input_adj_multi_list.append(input_adj_multi)
        input_img_whole_list.append(input_img_multi)
        input_adj_whole_list.append(input_adj_multi)
        pair_image_out= pair_image_model([input_img_multi,input_adj_multi])
        #pair_adj_out=adj_pair_image_model(input_adj_multi)
        #pair_image_out=keras.layers.concatenate([pair_image_out1,pair_adj_out],axis=-1)
        #pair_image_out=LWF.LWF(pair_image_out1,pair_adj_out)
        pair_image_out_list.append(pair_image_out)
        #pair_adj_out_list.append(pair_adj_out)
    merged_vector = keras.layers.concatenate(pair_image_out_list[:], axis=-1)  # modify this sentence to merge
    merged_model = keras.Model([input_img_multi_list,input_adj_multi_list], merged_vector)
    merged_out = merged_model([input_img_multi_list,input_adj_multi_list])
    combined_layer = keras.layers.concatenate([single_image_out, merged_out], axis=-1)
    combined_layer = keras.layers.Dropout(0.35)(combined_layer)
    combined_layer = keras.layers.Dropout(0.5)(combined_layer)

    combined = keras.layers.Dense(64, activation='relu')(combined_layer)
    combined = keras.layers.Dropout(0.3)(combined)
    combined = keras.layers.Dropout(0.5)(combined)

    #combined = keras.layers.Dense(32, activation='relu')(combined)
    #combined = keras.layers.Dropout(0.5)(combined)

    combined = keras.layers.Dense(16, activation='relu')(combined)
    #combined = keras.layers.Dropout(0.3)(combined)
    combined = keras.layers.Dropout(0.5)(combined)
    if num_classes < 2:
        print('no enough categories')
        sys.exit()
    elif num_classes == 2:
        combined = keras.layers.Dense(1, activation='sigmoid')(combined)
        #combined = keras.layers.Dense(2, activation='softmax')(combined)
        model = keras.Model([input_img_whole_list,input_adj_whole_list], combined)
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        adamw = keras.optimizers.Adam(lr=4e-3, beta_1=0.9, beta_2=0.999, decay=0.05)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    #early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    # patience：没有进步的训练轮数，在这之后训练就会被停止；verbose：详细信息模式；在auto模式下，从监测数量的名称自动推断方向。
    #checkpoint1 = ModelCheckpoint(filepath=dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                 # monitor='val_loss',
                                #  verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    #checkpoint2 = ModelCheckpoint(filepath=dir + '/weights.hdf5', monitor='val_accuracy', verbose=1,
                                #  save_best_only=True, mode='auto', period=1)
    #callbacks_list = [checkpoint2, early_stopping]
    return model

def smooth_curve(points, factor=0.8):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

def average_acc_loss(all_acc, all_val_acc, all_loss, all_val_loss):
    num = []
    for j in range(len(all_acc)):
        num.append(len(all_acc[j]))
    maxnum = max(num)

    # all_acc  all_val_acc  all_loss 的个数应该是一样的
    for h in range(len(all_acc)):
        #         print(all_acc[h])
        all_acc[h] = list(all_acc[h] + [0] * (maxnum - len(all_acc[h])))
        all_val_acc[h] = list(all_val_acc[h] + [0] * (maxnum - len(all_val_acc[h])))
        all_loss[h] = list(all_loss[h] + [0] * (maxnum - len(all_loss[h])))
        all_val_loss[h] = list(all_val_loss[h] + [0] * (maxnum - len(all_val_loss[h])))
        #         print(all_acc[h])
        all_acc[h] = np.array(all_acc[h])
        all_val_acc[h] = np.array(all_val_acc[h])
        all_loss[h] = np.array(all_loss[h])
        all_val_loss[h] = np.array(all_val_loss[h])

    all_acc = np.array(all_acc)
    all_val_acc = np.array(all_val_acc)
    all_loss = np.array(all_loss)
    all_val_loss = np.array(all_val_loss)

    # 求出平均值
    avg_acc = []
    avg_val_acc = []
    avg_loss = []
    avg_val_loss = []
    for g in range(len(all_acc[0])):
        b_acc = [i[g] for i in all_acc]
        b_val_acc = [i[g] for i in all_val_acc]
        b_loss = [i[g] for i in all_loss]
        b_val_loss = [i[g] for i in all_val_loss]
        print(b_acc)
        changdu = 0
        for bb in range(len(b_acc)):
            if b_acc[bb] != 0:
                changdu += 1

        avg_acc_s = np.sum(b_acc) / changdu
        avg_val_acc_s = np.sum(b_val_acc) / changdu
        avg_loss_s = np.sum(b_loss) / changdu
        avg_val_loss_s = np.sum(b_val_loss) / changdu

        avg_acc.append(avg_acc_s)
        avg_val_acc.append(avg_val_acc_s)
        avg_loss.append(avg_loss_s)
        avg_val_loss.append(avg_val_loss_s)
    epochs = range(1, len(avg_acc) + 1)

    return epochs, avg_acc, avg_val_acc, avg_loss, avg_val_loss