# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 21:04
# @Author  : zxl
# @FileName: main.py

import random

from MMOE.mmoe.mmoe import MMoE
import tensorflow as tf
from keras import backend as K
from keras import metrics
from keras.optimizers import Adam
from keras.initializers import VarianceScaling
from keras.layers import Input, Dense
from keras.models import Model


import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import hamming_loss,precision_score,f1_score,accuracy_score,recall_score,coverage_error,label_ranking_loss,average_precision_score,roc_auc_score
from sklearn.metrics import precision_score,recall_score
# from Predict.DataHandler import *
from DataUtil.DataHandler import *
from MTL.batchNN import nnMTL2
from MTL.batchNN2 import nnMTL3

"""
将gpa变成分类问题
"""
def SeperateGpa(gpa_y):
    """
    将gpa变成多类别问题
    :param gpa_y: array:(n,)
    :return:
    """

    new_y=[]
    for g in gpa_y:
        if g<2.5:
            new_y.append((1,0,0,0))
        elif g<3:
            new_y.append((0,1,0,0))
        elif  g<3.5:
            new_y.append((0,0,1,0))
        else:
            new_y.append((0,0,0,1))
    return np.array(new_y)

def SeperateFailed(failed_y):
    new_y=[]
    for f in failed_y:
        tmp=[0,0]
        tmp[int(f)]=1
        new_y.append(tmp)
    return np.array(new_y)



(train_X,train_Y,test_X,test_Y)=LoadData()

train_gpa_y=train_Y[:,0].flatten()
train_failed_y=train_Y[:,1].flatten()
test_gpa_y=test_Y[:,0].flatten()
test_failed_y=test_Y[:,1].flatten()
#将gpa变成多分类问题

test_gpa_y=SeperateGpa(test_gpa_y)
# train_failed_y=SeperateFailed(train_failed_y)
test_failed_y=SeperateFailed(test_failed_y)
# train_Y=np.array([train_gpa_y,train_failed_y])

validation_size=int(np.floor(len(test_failed_y)*0.5))
validation_gpa_y=test_gpa_y[:validation_size]
validation_failed_y=test_failed_y[:validation_size]
validation_Y=[validation_gpa_y,validation_failed_y]
validation_X=test_X[:validation_size]

test_gpa_y=test_gpa_y[validation_size:]
test_failed_y=test_failed_y[validation_size:]
test_X=test_X[validation_size:]
test_Y=[test_gpa_y,test_failed_y]



#对train中failed的样本进行重复采样
failed_idx=np.argwhere(train_failed_y==0).flatten()
dup_idx=np.random.choice(failed_idx,len(failed_idx)*3,replace=True)#有放回抽取
train_X=np.concatenate((train_X,train_X[dup_idx]),axis=0)
train_failed_y=np.append(train_failed_y,train_failed_y[dup_idx])
train_gpa_y=np.append(train_gpa_y,train_gpa_y[dup_idx])

train_failed_y=SeperateFailed(train_failed_y)
train_gpa_y=SeperateGpa(train_gpa_y)
train_Y=[train_gpa_y,train_failed_y]

print("Load finished!")
print("训练集记录条数:%d"%len(train_X))
print("测试集记录条数:%d"%len(test_X))
print("验证集记录条数:%d"%len(validation_X))
print("训练集中正负样本比例：%d:%d"%(len(np.argwhere(train_failed_y==0).flatten()),len(np.argwhere(train_failed_y==1).flatten())))



SEED = 1

# Fix numpy seed for reproducibility
np.random.seed(SEED)

# Fix random seed for reproducibility
random.seed(SEED)

# Fix TensorFlow graph-level seed for reproducibility
tf.set_random_seed(SEED)
tf_session = tf.Session(graph=tf.get_default_graph())
K.set_session(tf_session)


num_features=train_X.shape[1]


# Set up the input layer
input_layer = Input(shape=(num_features,))

# Set up MMoE layer
mmoe_layers = MMoE(
    units=32,
    num_experts=4,
    num_tasks=2
)(input_layer)

output_layers = []

output_info = ['y_gpa', 'y_failed']

# Build tower layer from MMoE layer
for index, task_layer in enumerate(mmoe_layers):
    tower_layer = Dense(
        units=8,
        activation='relu',
        kernel_initializer=VarianceScaling())(task_layer)
    if index==0:
        units_num=4
    else:
        units_num=2
    output_layer = Dense(
        units=units_num,
        name=output_info[index],
        activation='linear',
        kernel_initializer=VarianceScaling())(tower_layer)
    output_layers.append(output_layer)

# Compile model
model = Model(inputs=[input_layer], outputs=output_layers)
learning_rates = [1e-4, 1e-3, 1e-2]
adam_optimizer = Adam(lr=learning_rates[0])
model.compile(
    loss={'y_gpa': 'binary_crossentropy', 'y_failed': 'binary_crossentropy'},
    optimizer=adam_optimizer,
    metrics={'y_gpa':'accuracy','y_failed':'accuracy'}
)

# Print out model architecture summary
model.summary()

# Train the model
model.fit(
    x=train_X,
    y=train_Y,
    # batch_size=1000,
    validation_data=(validation_X, validation_Y),
    epochs=5000
)

# Test the model
res=model.predict(x=test_X)

predict_gpa_y=res[0]
predict_failed_y=res[1]
# for arr in res[0]:
#     max_idx=int(np.argmax(arr))
#     tmp=[0,0,0,0]
#     tmp[max_idx]=1
#     predict_gpa_y.append(tmp)
# for arr in res[1]:
#     if arr[0]>0.5:
#         predict_failed_y.append([1])
#     else:
#         predict_failed_y.append([0])


# print("----gpa回归预测-------")
# m=mean_squared_error(predict_gpa_y,test_gpa_y)
# print("mse:%f"%m)
# plt.scatter(test_gpa_y,predict_gpa_y)
# plt.plot(test_gpa_y,test_gpa_y,color="red")
# plt.show()
print("----gpa区间分类预测-------")
# f1=f1_score(test_gpa_y,predict_gpa_y,average='macro')
# print("f1_score:%f"%f1)
macro_auc=roc_auc_score(test_gpa_y,predict_gpa_y,average="macro")
print("macro auc:%f"%macro_auc)
# accuracy=accuracy_score(test_gpa_y,predict_gpa_y)
# print("accuracy:%f"%accuracy)
# precision=precision_score(test_gpa_y,predict_gpa_y,average="macro")
# print("precision:%f"%precision)
# recall=recall_score(test_gpa_y,predict_gpa_y,average="macro")
# print("recall:%f"%recall)


print("-------failed分类预测-------")


#只看挂科这一类分类效果好坏
# test_failed_y=[1-x for x in test_failed_y]#以是挂科了这个label作为评价指标的判断依据
# for item in predict_failed_y:
#     print(item)
for thres in [0.5]:
    # print("--------thres: -------------%f"%thres)
    predict_y2=predict_failed_y
    # predict_y2[predict_y2>=thres]=1
    # predict_y2[predict_y2<thres]=0
    # predict_failed_y=[1-x for x in predict_y2]
    # f1=f1_score(test_failed_y,predict_failed_y)
    # print("f1:%f"%f1)
    macro_auc=roc_auc_score(test_failed_y,predict_failed_y,average="macro")
    print("macro auc:%f"%macro_auc)
    # accuracy=accuracy_score(test_failed_y,predict_failed_y)
    # print("accuracy:%f"%accuracy)
    # precision=precision_score(test_failed_y,predict_failed_y)
    # print("precision:%f"%precision)
    # recall=recall_score(test_failed_y,predict_failed_y)
    # print("recall:%f"%recall)

print("挂科在评价时候用的是把通过当做真值")
