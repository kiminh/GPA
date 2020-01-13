# -*- coding: utf-8 -*-
# @Time    : 2019/11/24 15:59
# @Author  : zxl
# @FileName: MTLMain.py

from sklearn.metrics import f1_score,mean_squared_error,roc_auc_score
from sklearn.metrics import hamming_loss,precision_score,f1_score,accuracy_score,recall_score,coverage_error,label_ranking_loss,average_precision_score,roc_auc_score
from sklearn.metrics import precision_score,recall_score
from DataUtil.DataHandler import *
from MTL.Linear import LinearMTL

# (train_X,train_y,test_X,test_y)=LoadData2()
# train_gpa_y=train_y[:,0]
# train_failed_y=[]
# for item in train_y[:,1]:
#     if item:
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
#对train中failed的样本进行重复采样
# failed_idx=np.argwhere(train_failed_y==0).flatten()
# dup_idx=np.random.choice(failed_idx,len(failed_idx)*3,replace=True)#有放回抽取
# train_X=np.concatenate((train_X,train_X[dup_idx]),axis=0)
# train_failed_y=np.append(train_failed_y,train_failed_y[dup_idx])
# train_gpa_y=np.append(train_gpa_y,train_gpa_y[dup_idx])

# test_X=test_X.astype(np.float32)
# test_failed_y=test_failed_y.astype(np.float32)
# test_gpa_y=test_gpa_y.astype(np.float32)
#
# train_Y=np.array([train_gpa_y,train_failed_y])
(train_X,train_Y,test_X,test_Y)=LoadData()

train_gpa_y=train_Y[:,0].flatten()
train_failed_y=train_Y[:,1].flatten()
test_gpa_y=test_Y[:,0].flatten()
test_failed_y=test_Y[:,1].flatten()
train_Y=np.array([train_gpa_y,train_failed_y])

print("Load finished!")
print("训练集记录条数:%d"%len(train_X))
print("测试集记录条数:%d"%len(test_X))

print("训练集中正负样本比例：%d:%d"%(len(np.argwhere(train_failed_y==0).flatten()),len(np.argwhere(train_failed_y==1).flatten())))





model=LinearMTL()
model.fit(train_X,train_Y)
predict_Y=model.predict(test_X)
predict_gpa_y=predict_Y[0].flatten()
predict_y=predict_Y[1].flatten()

print(predict_gpa_y.shape)
print(test_gpa_y.shape)
print("----gpa回归预测-------")
m=mean_squared_error(predict_gpa_y,test_gpa_y)
print("mse:%f"%m)

print("-------failed分类预测-------")
test_failed_y=[1-x for x in test_failed_y]
for item in predict_y:
    print(item)
for thres in [0,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]:
    predict_y2=predict_y
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


