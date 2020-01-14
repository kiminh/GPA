# -*- coding: utf-8 -*-
# @Time    : 2020/1/13 9:12
# @Author  : zxl
# @FileName: main.py

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import hamming_loss,precision_score,f1_score,accuracy_score,recall_score,coverage_error,label_ranking_loss,average_precision_score,roc_auc_score
from sklearn.metrics import precision_score,recall_score
# from Predict.DataHandler import *
from DataUtil.DataHandler import *
from CVCTR.cvctr.CMTL import CMTL
from CVCTR.cvctr.GBDTMLP import gbdtmlp
from CVCTR.cvctr.CMTLR import CMTLR
from ModelEvaluation.Evaluation import Evaluate
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
        elif g<3.5:
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

validation_size=0
# validation_size=int(np.floor(len(test_failed_y)*0.5))
# validation_gpa_y=test_gpa_y[:validation_size]
# validation_failed_y=test_failed_y[:validation_size]
# validation_Y=[validation_gpa_y,validation_failed_y]
# validation_X=test_X[:validation_size]

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
# print("验证集记录条数:%d"%len(validation_X))
print("训练集中正负样本比例：%d:%d"%(len(np.argwhere(train_failed_y[:,0]==1).flatten()),len(np.argwhere(train_failed_y[:,1]==1).flatten())))

# model=CMTL()
# model=gbdtmlp()
model=CMTLR()
model.fit(train_X,train_Y)
predict_Y=model.predict(test_X)

predict_gpa=predict_Y[0]
predict_failed=predict_Y[1]

print("gpa 区间预测")
Evaluate(test_gpa_y,predict_gpa)

print("failed 预测")
Evaluate(test_failed_y,predict_failed)


