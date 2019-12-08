# -*- coding: utf-8 -*-
# @Time    : 2019/11/23 18:32
# @Author  : zxl
# @FileName: DataHandler.py

import json
import numpy as np
import pandas as pd

"""
将训练集合测试集数据处理成模型输入
类别数据：one-hot表示等
"""

def LoadData():
    root="C://zxl/Data/GPA/"
    train_root=root+"train/"
    test_root=root+"test/"
    train_stu_path=train_root+"train_stu.csv"
    train_gpa_path=train_root+"gpa.csv"
    test_stu_path=test_root+"test_stu.csv"
    test_gpa_path=test_root+"gpa.csv"
    train_entropy_path=train_root+"entropy.csv"
    test_entropy_path=test_root+"entropy.csv"

    train_stu_df=pd.read_csv(train_stu_path)
    train_gpa_df=pd.read_csv(train_gpa_path)
    test_stu_df=pd.read_csv(test_stu_path)
    test_gpa_df=pd.read_csv(test_gpa_path)
    train_entropy_df=pd.read_csv(train_entropy_path)
    test_entropy_df=pd.read_csv(test_entropy_path)

    (train_X,train_y)=ConstructData2(train_stu_df,train_gpa_df,train_entropy_df)
    (test_X,test_y)=ConstructData2(test_stu_df,test_gpa_df,test_entropy_df)

    return (train_X,train_y,test_X,test_y)

def ConstructData2(stu_df,gpa_df,entropy_df):
    """
    每个人只有一条记录待预测
    :param stu_df:
    :param gpa_df:
    :return:
    """

    stu_df = LoadStu(stu_df)  # one-hot表示

    gpa_df = LoadCourse(gpa_df)  # one-hot表示


    

    target_df = gpa_df[gpa_df['semester'] == 5]  # 最后一学期为待遇测内容
    history_df = gpa_df[gpa_df['semester'] != 5]  # 前几学期为历史

    history_info_df = LoadHistory(history_df)  # 前几学期的平均绩点

    target_df=LoadTarget(target_df)
    target_info_df = target_df[['stu_id','total_credit']]
    target_gpa_df = target_df[['stu_id', 'gpa', 'failed']]

    df = pd.merge(stu_df, history_info_df, on='stu_id')
    df = pd.merge(df, target_info_df, on='stu_id')
    df = pd.merge(df, entropy_df, on="stu_id")
    df = pd.merge(df, target_gpa_df, on='stu_id')




    X = np.array(df.ix[:, 1:-2])
    y = np.array(df.ix[:, -2:])
    return X, y

def ConstructData(stu_df,gpa_df):
    stu_df = LoadStu(stu_df)  # one-hot表示

    gpa_df = LoadCourse(gpa_df)  # one-hot表示
    column_names = gpa_df.columns.tolist()
    info_name_lst = ['stu_id','credit', 'stu_num']
    for name in column_names:
        if name.startswith('cat'):
            info_name_lst.append(name)

    target_df = gpa_df[gpa_df['semester'] == 5]  # 最后一学期为待遇测内容
    
    
    
    history_df = gpa_df[gpa_df['semester'] != 5]  # 前几学期为历史

    history_info_df = LoadHistory(history_df)  # 前几学期的平均绩点
    target_info_df = target_df[info_name_lst]
    target_gpa_df = target_df[['stu_id','gpa', 'failed']]

    df=pd.merge(stu_df,history_info_df,on='stu_id')
    df=pd.merge(df,target_info_df,on='stu_id')

    df=pd.merge(df,target_gpa_df,on='stu_id')

    X=np.array(df.ix[:,1:-2])
    y=np.array(df.ix[:,-2:])
    return X,y



def LoadTarget(target_df):
    """
    算一下总学分，以及综合绩点和是否挂科
    :param target_df:
    :return:
    """

    stu_lst=[]
    credit_lst=[]
    gpa_lst=[]
    failed_lst=[]
    for stu_id, sub_group in target_df.groupby('stu_id'):
        stu_lst.append(stu_id)
        total_credit=0.0
        total_gpa=0.0
        failed=False
        for credit,gpa,f in zip(sub_group.credit,sub_group.gpa,sub_group.failed):
            if f:#挂科的课程不计入计算
                failed=True
                continue
            total_credit+=credit
            total_gpa+=gpa*credit
        if total_credit==0:
            avg_gpa=0
        else:
            avg_gpa=total_gpa/total_credit
        credit_lst.append(total_credit)
        gpa_lst.append(avg_gpa)
        failed_lst.append(failed)
    df=pd.DataFrame({'stu_id':stu_lst,'total_credit':credit_lst,'gpa':gpa_lst,'failed':failed_lst})
    return df

def LoadStu(stu_df):
    """
    将性别、院系、入学时间处理成one-hot形式
    :param stu_df: 新的df
    :return:
    """
    new_stu_df = pd.get_dummies(stu_df, columns=["gender","dep","enroll_time"])
    return new_stu_df

def LoadCourse(course_df):
    """
    将课程类别处理成one-hot形式
    :param course_df:
    :return:
    """
    new_course_df=pd.get_dummies(course_df,columns=["cat"],prefix='cat')
    return new_course_df

def LoadHistory(history_df):
    """
    计算学生前几个学期的均绩、是否挂科
    :param history_df:
    :return: df，stu_id,semester,avg_gpa,failed
    """
    #一种处理方式是返回这个人前几个学期平均绩点，是否挂科,8维特征
    stu_lst=[]
    semester_lst=[]
    avg_gpa_lst=[]
    failed_lst=[]
    for group_name,gpa_df in history_df.groupby(['stu_id','semester']):
        stu_lst.append(group_name[0])
        semester_lst.append(group_name[1])
        failed=False
        avg_gpa=0.0
        total_credit=0
        for credit,gpa in zip(gpa_df.credit,gpa_df.gpa):
            if gpa==0:
                failed=True
                continue
            avg_gpa+=gpa
            total_credit+=credit
        if total_credit==0:
            avg_gpa=0
        else:
            avg_gpa/=total_credit
        avg_gpa_lst.append(avg_gpa)
        failed_lst.append(failed)
    avg_gpa_df=pd.DataFrame({'stu_id':stu_lst,'semester':semester_lst,'avg_gpa':avg_gpa_lst,'failed':failed_lst})

    stu_lst=[]
    gpa_m=[[],[],[],[]]
    failed_m=[[],[],[],[]]
    for stu_id,gpa_df in avg_gpa_df.groupby('stu_id'):
        stu_lst.append(stu_id)
        for i in range(4):
            gpa_m[i].append(gpa_df[gpa_df['semester']==i+1]['avg_gpa'].values[0])
            failed_m[i].append(gpa_df[gpa_df['semester']==i+1]['failed'].values[0])
    gpa_m=np.array(gpa_m)
    failed_m=np.array(failed_m)
    res_df=pd.DataFrame({'stu_id':stu_lst,'g1':gpa_m[0],'f1':failed_m[0],'g2':gpa_m[1],'f2':failed_m[1],
                         'g3':gpa_m[2],'f3':failed_m[2],'g4':gpa_m[3],'f4':failed_m[3]})
    return res_df

def LoadData2():
    root = "C:\\zxl\Data\GPA\\"
    train_root = root + "train\\"
    test_root = root + "test\\"

    trainX_save_path = train_root + "trainX2.npy"
    trainy_save_path = train_root + "trainy2.npy"
    testX_save_path = test_root + "testX2.npy"
    testy_save_path = test_root + "testy2.npy"

    train_X=np.load(trainX_save_path)
    train_y=np.load(trainy_save_path)
    test_X=np.load(testX_save_path)
    test_y=np.load(testy_save_path)
    return (train_X,train_y,test_X,test_y)

if __name__=="__main__":
    (train_X, train_y, test_X, test_y)=LoadData()
    root = "C:\\zxl\Data\GPA\\"
    train_root = root + "train\\"
    test_root = root + "test\\"

    trainX_save_path=train_root+"trainX2.npy"
    trainy_save_path=train_root+"trainy2.npy"
    testX_save_path=test_root+"testX2.npy"
    testy_save_path=test_root+"testy2.npy"

    np.save(trainX_save_path,train_X)
    np.save(trainy_save_path,train_y)
    np.save(testX_save_path,test_X)
    np.save(testy_save_path,test_y)
