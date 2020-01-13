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


random_state=5

(train_X,train_y,test_X,test_y)=LoadData()
# train_gpa_y=train_y[:,0]
# train_failed_y=[]
# for item in train_y[:,1]:
#     if item==True:
#         train_failed_y.append(0)
#     else:
#         train_failed_y.append(1)
# train_failed_y=np.array(train_failed_y)
# test_gpa_y=test_y[:,0]
# test_failed_y=[]
# for item in test_y[:,1]:
#     if item:
#         test_failed_y.append(0)
#     else:
#         test_failed_y.append(1)
# test_failed_y=np.array(test_failed_y)
#
# train_X=train_X.astype(np.float32)
# train_failed_y=train_failed_y.astype(np.float32)
# train_gpa_y=train_gpa_y.astype(np.float32)
# #对train中failed的样本进行重复采样
# failed_idx=np.argwhere(train_failed_y==0).flatten()
# dup_idx=np.random.choice(failed_idx,len(failed_idx)*3,replace=True)#有放回抽取
# train_X=np.concatenate((train_X,train_X[dup_idx]),axis=0)
# train_failed_y=np.append(train_failed_y,train_failed_y[dup_idx])
# train_gpa_y=np.append(train_gpa_y,train_gpa_y[dup_idx])
#
# test_X=test_X.astype(np.float32)
# test_failed_y=test_failed_y.astype(np.float32)
# test_gpa_y=test_gpa_y.astype(np.float32)




train_gpa_y=train_y[:,0].flatten()
train_failed_y=train_y[:,1].flatten()
test_gpa_y=test_y[:,0].flatten()
test_failed_y=test_y[:,1].flatten()

#划分验证集和测试集

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

print("Load finished!")
print("训练集记录条数:%d"%len(train_X))
print("测试集记录条数:%d"%len(test_X))

print("训练集中正负样本比例：%d:%d"%(len(np.argwhere(train_failed_y==0).flatten()),len(np.argwhere(train_failed_y==1).flatten())))







"""
是否挂科
"""
print("是否挂科 二分类任务")

#只看挂科这一类分类效果好坏
test_failed_y=[1-x for x in test_failed_y]


model=MLPClassifier()
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
#只看挂科这一类分类效果好 坏
predict_failed_y=[1-x for x in predict_y]
print("------MLP---------")
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



model=DecisionTreeClassifier()
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
#只看挂科这一类分类效果好 坏
predict_failed_y=[1-x for x in predict_y]
print("------DT---------")
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

model=neighbors.KNeighborsClassifier()
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
#只看挂科这一类分类效果好 坏
predict_failed_y=[1-x for x in predict_y]
print("------KNN---------")
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

model=ensemble.RandomForestClassifier(n_estimators=20,random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
#只看挂科这一类分类效果好 坏
predict_failed_y=[1-x for x in predict_y]
print("------随机森林---------")
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

model=ensemble.GradientBoostingClassifier(n_estimators=20,random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
#只看挂科这一类分类效果好 坏
predict_failed_y=[1-x for x in predict_y]
print("------GBDT---------")
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

model=ensemble.BaggingClassifier(random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
#只看挂科这一类分类效果好 坏
predict_failed_y=[1-x for x in predict_y]
print("------Bagging---------")
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

model=tree.ExtraTreeClassifier(random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
#只看挂科这一类分类效果好 坏
predict_failed_y=[1-x for x in predict_y]
print("------ExtraTree---------")
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

model=ensemble.AdaBoostClassifier(n_estimators=20,random_state=random_state)
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
#只看挂科这一类分类效果好 坏
predict_failed_y=[1-x for x in predict_y]
print("------Adaboost---------")
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

model=svm.SVC()
predict_y=model.fit(train_X,train_failed_y).predict(test_X)
#只看挂科这一类分类效果好 坏
predict_failed_y=[1-x for x in predict_y]
print("------SVC---------")
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

"""
gpa
"""
print("---------gpa 回归任务-----------------")

model=linear_model.LinearRegression()
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("Linear：%f"%mse)
# plt.scatter(test_gpa_y,predict_y)
# plt.show()

model=MLPRegressor()
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
m=mean_squared_error(test_gpa_y,predict_y)
print("MLP:%f"%m)

plt.scatter(test_gpa_y,predict_y)
plt.show()



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

model=ensemble.AdaBoostRegressor(n_estimators=50,random_state=random_state)
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("Adaboost：%f"%mse)

model=svm.SVR(C=10)
predict_y=model.fit(train_X,train_gpa_y).predict(test_X)
mse=mean_squared_error(test_gpa_y,predict_y)
print("SVC：%f"%mse)