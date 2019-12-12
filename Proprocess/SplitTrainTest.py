# -*- coding: utf-8 -*-
# @Time    : 2019/12/9 17:02
# @Author  : zxl
# @FileName: SplitTrainTest.py

import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


"""
将数据处理为
(20152,20161)
(20161,20162)
.....等pair
并且把数据分为train和test
"""



def SplitTrainTest(df,trainX_path,trainY_path,testX_path,testY_path):
    """
    将数据划分为train和test，并且保存到文件
    :param df:
    :param trainX_path:
    :param trainY_path:
    :param testX_path:
    :param testY_path:
    :return:
    """
    stu_lst=list(set(df.stu_id.values))
    train_set=set(random.sample(stu_lst, int(len(stu_lst) * 0.7)))
    test_set=set(stu_lst).difference(train_set)

    train_df=df[df.stu_id.isin(train_set)]
    test_df=df[df.stu_id.isin(test_set)]

    trainY=np.array(train_df[['gpa','failed']].ix[:,:])
    trainX=np.array(train_df.drop(['stu_id', 'gpa', 'failed','lib_num','breakfast_num','lunch_num','dinner_num',
                                   'post_lib_num', 'post_breakfast_num','post_lunch_num','post_dinner_num'],axis=1).ix[:,:])
    testY = np.array(test_df[['gpa', 'failed']].ix[:,:])
    testX = np.array(test_df.drop(['stu_id', 'gpa', 'failed','lib_num','breakfast_num','lunch_num','dinner_num',
                                   'post_lib_num', 'post_breakfast_num','post_lunch_num','post_dinner_num'],axis=1).ix[:,:])

    scaler = MinMaxScaler()
    trainX = scaler.fit_transform(trainX)
    scaler = MinMaxScaler()
    testX = scaler.fit_transform(testX)

    print("train number: %d "%len(train_set))
    print("test number: %d"%len(test_set))
    print("train record number: %d"%len(trainX))
    print("test record number: %d"%len(testX))
    np.save(trainX_path,trainX)
    np.save(trainY_path,trainY)
    np.save(testX_path,testX)
    np.save(testY_path,testY)



if __name__=="__main__":
    root="C://zxl/Data/GPA-large/"
    consume_file=root+"processed/consume2.csv"
    gpa_file=root+"processed/gpa.csv"
    lib_file=root+"processed/lib2.csv"
    profile_file=root+"stu/profile.csv"
    out_path=root+"processed/pair2.csv"
    train_root=root+"train/"
    test_root=root+"test/"
    trainX_path=train_root+"trainX2.npy"
    trainY_path=train_root+"trainY2.npy"
    testX_path=test_root+"testX2.npy"
    testY_path=test_root+"testY2.npy"


    consume_df=pd.read_csv(consume_file,index_col=0)
    gpa_df=pd.read_csv(gpa_file)

    lib_df=pd.read_csv(lib_file)
    profile_df=pd.read_csv(profile_file)

    # PairData(consume_df,gpa_df,lib_df,profile_df,out_path)

    df=pd.read_csv(out_path)
    SplitTrainTest(df, trainX_path, trainY_path, testX_path, testY_path)



