# -*- coding: utf-8 -*-
# @Time    : 2019/11/23 21:11
# @Author  : zxl
# @FileName: sklearnMain.py

from sklearn import linear_model
from sklearn import svm,neighbors,ensemble,tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,mean_squared_error

from Predict.DataHandler import *

random_state=5

(train_X,train_y,test_X,test_y)=LoadData()
train_gpa_y=train_y[:,0]
train_failed_y=[]
for item in train_y[:,1]:
    if item==True:
        train_failed_y.append(1)
    else:
        train_failed_y.append(0)
train_failed_y=np.array(train_failed_y)
test_gpa_y=test_y[:,0]
test_failed_y=[]
for item in test_y[:,1]:
    if item:
        test_failed_y.append(1)
    else:
        test_failed_y.append(0)
test_failed_y=np.array(test_failed_y)
print("Load finished!")
print("训练集记录条数:%d"%len(train_X))
print("测试集记录条数:%d"%len(test_X))
print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)
print(train_failed_y.shape)
print(train_gpa_y.shape)
print(test_failed_y.shape)
print(test_gpa_y.shape)
"""
是否挂科
"""

model=DecisionTreeClassifier()
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
print("决策树：%f"%f1)

model=ensemble.RandomForestClassifier(n_estimators=20,random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
print("随机森林：%f"%f1)

model=neighbors.KNeighborsClassifier()
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
print("KNN：%f"%f1)

model=ensemble.GradientBoostingClassifier(n_estimators=20,random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
print("GBRT：%f"%f1)

model=ensemble.AdaBoostClassifier(n_estimators=20,random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
print("Adaboost：%f"%f1)


model=ensemble.BaggingClassifier(random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
print("Bagging：%f"%f1)

model=tree.ExtraTreeClassifier(random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
print("ExtraTree：%f"%f1)

model=svm.SVC()
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
f1=f1_score(test_failed_y,predict_y)
print("SVM：%f"%f1)
"""
gpa
"""
model=tree.DecisionTreeRegressor(random_state=1)
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("ExtraTree：%f"%mse)

model=linear_model.LinearRegression()
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("Linear：%f"%mse)

model=neighbors.KNeighborsRegressor()
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("KNN：%f"%mse)

model=ensemble.RandomForestRegressor(n_estimators=20,random_state=1)
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("随机森林：%f"%mse)

model=ensemble.AdaBoostRegressor(n_estimators=50,random_state=random_state)
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("Adaboost：%f"%mse)

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


model=svm.SVR(C=10)
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("SVC：%f"%mse)