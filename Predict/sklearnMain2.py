# -*- coding: utf-8 -*-
# @Time    : 2019/11/23 21:11
# @Author  : zxl
# @FileName: sklearnMain.py

from sklearn import linear_model
from sklearn import svm,neighbors,ensemble,tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import hamming_loss,precision_score,f1_score,accuracy_score,recall_score,coverage_error,label_ranking_loss,average_precision_score,roc_auc_score
from sklearn.neural_network import MLPRegressor,MLPClassifier

# from Predict.DataHandler import *
from DataUtil.DataHandler import *
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
import matplotlib.pyplot as plt
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
        elif  g<3:
            new_y.append((0,1,0,0))
        elif  g<3.5:
            new_y.append((0,0,1,0))
        else:
            new_y.append((0,0,0,1))
    return np.array(new_y)


random_state=5

(train_X,train_y,test_X,test_y)=LoadData()

train_gpa_y=train_y[:,0].flatten()
train_failed_y=train_y[:,1].flatten()
test_gpa_y=test_y[:,0].flatten()
test_failed_y=test_y[:,1].flatten()


test_gpa_y=SeperateGpa(test_gpa_y)

#划分验证集和测试集
validation_size=0
# validation_size=int(np.floor(len(test_failed_y)*0.5))
# validation_gpa_y=test_gpa_y[:validation_size]
# validation_failed_y=test_failed_y[:validation_size]
# validation_Y=[np.reshape(validation_gpa_y,newshape=(-1,1)),np.reshape(validation_failed_y,newshape=(-1,1))]
# validation_X=test_X[:validation_size]

test_gpa_y=test_gpa_y[validation_size:]
test_failed_y=test_failed_y[validation_size:]
test_X=test_X[validation_size:]
test_Y=[test_gpa_y,np.reshape(test_failed_y,newshape=(-1,1))]



#对train中failed的样本进行重复采样
train_gpa_y=train_gpa_y.flatten()
failed_idx=np.argwhere(train_failed_y==0).flatten()
dup_idx=np.random.choice(failed_idx,len(failed_idx)*3,replace=True)#有放回抽取
train_X=np.concatenate((train_X,train_X[dup_idx]),axis=0)
train_failed_y=np.append(train_failed_y,train_failed_y[dup_idx])
train_gpa_y=np.append(train_gpa_y,train_gpa_y[dup_idx])
train_gpa_y=SeperateGpa(train_gpa_y)

print("Load finished!")
print("训练集记录条数:%d"%len(train_X))
print("测试集记录条数:%d"%len(test_X))

print("训练集中正负样本比例：%d:%d"%(len(np.argwhere(train_failed_y==0).flatten()),len(np.argwhere(train_failed_y==1).flatten())))

print("是否挂科 二分类任务")

#只看挂科这一类分类效果好坏
# test_failed_y=[1-x for x in test_failed_y]

new_train_failed_y=[]
for val in train_failed_y:
    tmp=[0,0]
    tmp[int(val)]=1
    new_train_failed_y.append(tmp)
new_train_failed_y=np.array(new_train_failed_y)
# train_failed_y=new_train_failed_y
new_test_failed_y=[]
for val in test_failed_y:
    tmp=[0,0]
    tmp[int(val)]=1
    new_test_failed_y.append(tmp)
new_test_failed_y=np.array(new_test_failed_y)
# test_failed_y=np.array(new_test_failed_y)



model=svm.SVC(probability=True)
predict_y=model.fit(train_X,train_failed_y).predict_proba(test_X)
print("------SVC---------")
Evaluate(new_test_failed_y,predict_y)



model=MLPClassifier()
predict_y=model.fit(train_X,new_train_failed_y).predict_proba(test_X)
print("------MLP---------")
Evaluate(new_test_failed_y,predict_y)



model=DecisionTreeClassifier()
predict_y=model.fit(train_X,train_failed_y).predict_proba(test_X)
print("------DT---------")
Evaluate(new_test_failed_y,predict_y)


model=neighbors.KNeighborsClassifier()
predict_y=model.fit(train_X,train_failed_y).predict_proba(test_X)
print("------KNN---------")
Evaluate(new_test_failed_y,predict_y)


model=ensemble.RandomForestClassifier(n_estimators=20,random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict_proba(test_X)
print("------随机森林---------")
Evaluate(new_test_failed_y,predict_y)

model=ensemble.GradientBoostingClassifier(n_estimators=20,random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict_proba(test_X)
print("------GBDT---------")
Evaluate(new_test_failed_y,predict_y)

model=ensemble.BaggingClassifier(random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict_proba(test_X)
print("------Bagging---------")
Evaluate(new_test_failed_y,predict_y)

model=tree.ExtraTreeClassifier(random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict_proba(test_X)
print("------ExtraTree---------")
Evaluate(new_test_failed_y,predict_y)

model=ensemble.AdaBoostClassifier(n_estimators=20,random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict_proba(test_X)
print("------Adaboost---------")
Evaluate(new_test_failed_y,predict_y)


print("---------gpa 区间预测任务-----------------")

new_train_gpa_y=[]

new_test_gpa_y=[]
for arr in train_gpa_y:
    idx=np.argwhere(arr==1)[0][0]
    new_train_gpa_y.append(idx)
new_train_gpa_y=np.array(new_train_gpa_y)

model=MLPClassifier()
predict_y=model.fit(train_X,new_train_gpa_y).predict_proba(test_X)
m=roc_auc_score(test_gpa_y,predict_y,average='macro')
print("MLP:%f"%m)

model=tree.DecisionTreeClassifier()
predict_y=model.fit(train_X,new_train_gpa_y).predict_proba(test_X)
m=roc_auc_score(test_gpa_y,predict_y,average='macro')
print("决策树:%f"%m)

model=neighbors.KNeighborsClassifier()
predict_y=model.fit(train_X,new_train_gpa_y).predict_proba(test_X)
mse=roc_auc_score(test_gpa_y,predict_y,average='macro')
print("KNN：%f"%mse)

model=ensemble.RandomForestClassifier(n_estimators=20,random_state=1)
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=roc_auc_score(test_gpa_y,predict_y,average='macro')
print("随机森林：%f"%mse)



model=ensemble.GradientBoostingClassifier(n_estimators=100,random_state=1)
predict_y=model.fit(train_X,new_train_gpa_y).predict_proba(test_X)
mse=roc_auc_score(test_gpa_y,predict_y,average='macro')
print("GBRT：%f"%mse)

model=ensemble.BaggingClassifier(random_state=1)
predict_y=model.fit(train_X,new_train_gpa_y).predict_proba(test_X)
mse=roc_auc_score(test_gpa_y,predict_y,average='macro')
print("Bagging：%f"%mse)


model=tree.ExtraTreeClassifier(random_state=1)
predict_y=model.fit(train_X,new_train_gpa_y).predict_proba(test_X)
mse=roc_auc_score(test_gpa_y,predict_y,average='macro')
print("ExtraTree：%f"%mse)

model=ensemble.AdaBoostClassifier(n_estimators=50,random_state=random_state)
predict_y=model.fit(train_X,new_train_gpa_y).predict_proba(test_X)
mse=roc_auc_score(test_gpa_y,predict_y,average='macro')
print("Adaboost：%f"%mse)

model=svm.SVC(C=10,probability=True)
predict_y=model.fit(train_X,new_train_gpa_y).predict_proba(test_X)
mse=roc_auc_score(test_gpa_y,predict_y,average='macro')
print("SVC：%f"%mse)