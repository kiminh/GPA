# -*- coding: utf-8 -*-
# @Time    : 2019/11/23 21:11
# @Author  : zxl
# @FileName: sklearnMain.py

from sklearn import linear_model
from sklearn import svm,neighbors,ensemble,tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,mean_squared_error,roc_auc_score
from sklearn.neural_network import MLPRegressor,MLPClassifier

from Predict.DataHandler import *

random_state=5

(train_X,train_y,test_X,test_y)=LoadData2()
train_gpa_y=train_y[:,0]
train_failed_y=[]
for item in train_y[:,1]:
    if item==True:
        train_failed_y.append(0)
    else:
        train_failed_y.append(1)
train_failed_y=np.array(train_failed_y)
test_gpa_y=test_y[:,0]
test_failed_y=[]
for item in test_y[:,1]:
    if item:
        test_failed_y.append(0)
    else:
        test_failed_y.append(1)
test_failed_y=np.array(test_failed_y)

train_X=train_X.astype(np.float32)
train_failed_y=train_failed_y.astype(np.float32)
train_gpa_y=train_gpa_y.astype(np.float32)
#对train中failed的样本进行重复采样
# failed_idx=np.argwhere(train_failed_y==0).flatten()
# dup_idx=np.random.choice(failed_idx,len(failed_idx)*3,replace=True)#有放回抽取
# train_X=np.concatenate((train_X,train_X[dup_idx]),axis=0)
# train_failed_y=np.append(train_failed_y,train_failed_y[dup_idx])
# train_gpa_y=np.append(train_gpa_y,train_gpa_y[dup_idx])

test_X=test_X.astype(np.float32)
test_failed_y=test_failed_y.astype(np.float32)
test_gpa_y=test_gpa_y.astype(np.float32)



print("Load finished!")
print("训练集记录条数:%d"%len(train_X))
print("测试集记录条数:%d"%len(test_X))

print("训练集中正负样本比例：%d:%d"%(len(np.argwhere(train_failed_y==0).flatten()),len(np.argwhere(train_failed_y==1).flatten())))







"""
是否挂科
"""
print("是否挂科 二分类任务")

model=MLPClassifier()
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
macro_auc=roc_auc_score(test_failed_y,predict_y,average="macro")
micro_auc=roc_auc_score(test_failed_y,predict_y,average="micro")
print("MLP-- f1:%f, macro:%f, micro:%f"%(f1,macro_auc,micro_auc))


model=DecisionTreeClassifier()
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
macro_auc=roc_auc_score(test_failed_y,predict_y,average="macro")
micro_auc=roc_auc_score(test_failed_y,predict_y,average="micro")
print("决策树-- f1:%f, macro:%f, micro:%f"%(f1,macro_auc,micro_auc))

model=neighbors.KNeighborsClassifier()
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
macro_auc=roc_auc_score(test_failed_y,predict_y,average="macro")
micro_auc=roc_auc_score(test_failed_y,predict_y,average="micro")
print("KNN-- f1:%f, macro:%f, micro:%f"%(f1,macro_auc,micro_auc))

model=ensemble.RandomForestClassifier(n_estimators=20,random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
macro_auc=roc_auc_score(test_failed_y,predict_y,average="macro")
micro_auc=roc_auc_score(test_failed_y,predict_y,average="micro")
print("随机森林-- f1:%f, macro:%f, micro:%f"%(f1,macro_auc,micro_auc))

model=ensemble.GradientBoostingClassifier(n_estimators=20,random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
macro_auc=roc_auc_score(test_failed_y,predict_y,average="macro")
micro_auc=roc_auc_score(test_failed_y,predict_y,average="micro")
print("GBRT-- f1:%f, macro:%f, micro:%f"%(f1,macro_auc,micro_auc))

model=ensemble.BaggingClassifier(random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
macro_auc=roc_auc_score(test_failed_y,predict_y,average="macro")
micro_auc=roc_auc_score(test_failed_y,predict_y,average="micro")
print("Bagging-- f1:%f, macro:%f, micro:%f"%(f1,macro_auc,micro_auc))

model=tree.ExtraTreeClassifier(random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
macro_auc=roc_auc_score(test_failed_y,predict_y,average="macro")
micro_auc=roc_auc_score(test_failed_y,predict_y,average="micro")
print("ExtraTree-- f1:%f, macro:%f, micro:%f"%(f1,macro_auc,micro_auc))

# model=ensemble.AdaBoostClassifier(n_estimators=20,random_state=random_state)
# predict_y=model.fit(train_X,train_failed_y).predict(test_X)
# f1=f1_score(test_failed_y,predict_y)
# print("Adaboost：%f"%f1)

# model=svm.SVC()
# predict_y=model.fit(train_X,train_failed_y).predict(test_X)
# f1=f1_score(test_failed_y,predict_y)
# print("SVM：%f"%f1)
"""
gpa
"""
print("---------gpa 回归任务-----------------")

model=MLPRegressor()
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
m=mean_squared_error(test_gpa_y,predict_y)
print("MLP:%f"%m)

model=linear_model.LinearRegression()
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("Linear：%f"%mse)

model=tree.DecisionTreeRegressor()
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
m=mean_squared_error(test_gpa_y,predict_y)
print("决策树:%f"%m)

model=neighbors.KNeighborsRegressor()
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("KNN：%f"%mse)

model=ensemble.RandomForestRegressor(n_estimators=20,random_state=1)
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("随机森林：%f"%mse)

model=ensemble.GradientBoostingRegressor(n_estimators=100,random_state=1)
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("GBRT：%f"%mse)

model=ensemble.BaggingRegressor(random_state=1)
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("Bagging：%f"%mse)

model=tree.ExtraTreeRegressor(random_state=1)
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("ExtraTree：%f"%mse)

# model=ensemble.AdaBoostRegressor(n_estimators=50,random_state=random_state)
# predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
# mse=mean_squared_error(test_gpa_y,predict_y)
# print("Adaboost：%f"%mse)

# model=svm.SVR(C=10)
# predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
# mse=mean_squared_error(test_gpa_y,predict_y)
# print("SVC：%f"%mse)