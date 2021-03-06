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





(train_X,train_Y,test_X,test_Y)=LoadData()

train_gpa_y=train_Y[:,0].flatten()
train_failed_y=train_Y[:,1].flatten()
test_gpa_y=test_Y[:,0].flatten()
test_failed_y=test_Y[:,1].flatten()

# train_Y=np.array([train_gpa_y,train_failed_y])

validation_size=int(np.floor(len(test_failed_y)*0.5))
validation_gpa_y=test_gpa_y[:validation_size]
validation_failed_y=test_failed_y[:validation_size]
validation_Y=[np.reshape(validation_gpa_y,newshape=(-1,1)),np.reshape(validation_failed_y,newshape=(-1,1))]
validation_X=test_X[:validation_size]

test_gpa_y=test_gpa_y[validation_size:]
test_failed_y=test_failed_y[validation_size:]
test_X=test_X[validation_size:]
test_Y=[np.reshape(test_gpa_y,newshape=(-1,1)),np.reshape(test_failed_y,newshape=(-1,1))]


#对train中failed的样本进行重复采样
# failed_idx=np.argwhere(train_failed_y==0).flatten()
# dup_idx=np.random.choice(failed_idx,len(failed_idx)*3,replace=True)#有放回抽取
# train_X=np.concatenate((train_X,train_X[dup_idx]),axis=0)
# train_failed_y=np.append(train_failed_y,train_failed_y[dup_idx])
# train_gpa_y=np.append(train_gpa_y,train_gpa_y[dup_idx])


train_Y=[np.reshape(train_gpa_y,newshape=(-1,1)),np.reshape(train_failed_y,newshape=(-1,1))]

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
    units=16,
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
    output_layer = Dense(
        units=1,
        name=output_info[index],
        activation='linear',
        kernel_initializer=VarianceScaling())(tower_layer)
    output_layers.append(output_layer)

# Compile model
model = Model(inputs=[input_layer], outputs=output_layers)
learning_rates = [1e-4, 1e-3, 1e-2]
adam_optimizer = Adam(lr=learning_rates[0])
model.compile(
    loss={'y_gpa': 'mse', 'y_failed': 'binary_crossentropy'},
    optimizer=adam_optimizer,
    metrics={'y_gpa':'mse','y_failed':'accuracy'}
)

# Print out model architecture summary
model.summary()

# Train the model
model.fit(
    x=train_X,
    y=train_Y,
    # batch_size=32,
    validation_data=(validation_X, validation_Y),
    epochs=1
)

# Test the model
res=model.predict(x=test_X)
res=np.array(res)

predict_gpa_y=res[0].flatten()
predict_failed_y=res[1].flatten()

print("----gpa回归预测-------")
m=mean_squared_error(predict_gpa_y,test_gpa_y)
print("mse:%f"%m)
plt.scatter(test_gpa_y,predict_gpa_y)
plt.plot(test_gpa_y,test_gpa_y,color="red")
plt.show()


print("-------failed分类预测-------")
print(set(test_failed_y))

#只看挂科这一类分类效果好坏
test_failed_y=[1-x for x in test_failed_y]#以是挂科了这个label作为评价指标的判断依据
# for item in predict_failed_y:
#     print(item)
for thres in [0.5]:
    predict_y2=predict_failed_y
    predict_y2[predict_y2>=thres]=1
    predict_y2[predict_y2<thres]=0
    predict_failed_y=[1-x for x in predict_y2]
    print("---------------thres:%f-----------"%thres)
    f1=f1_score(test_failed_y,predict_failed_y)
    print("f1:%f"%f1)
    macro_auc=roc_auc_score(test_failed_y,predict_failed_y,average="macro")
    print("macro auc:%f"%macro_auc)
    accuracy=accuracy_score(test_failed_y,predict_failed_y)
    print("accuracy:%f"%accuracy)
    precision=precision_score(test_failed_y,predict_failed_y)
    print("precision:%f"%precision)
    recall=recall_score(test_failed_y,predict_failed_y)
    print("recall:%f"%recall)


