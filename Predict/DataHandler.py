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

    train_stu_df=pd.read_csv(train_stu_path)
    train_gpa_df=pd.read_csv(train_gpa_path)
    test_stu_df=pd.read_csv(test_stu_path)
    test_gpa_df=pd.read_csv(test_gpa_path)

    (train_X,train_y)=ConstructData(train_stu_df,train_gpa_df)
    (test_X,test_y)=ConstructData(test_stu_df,test_gpa_df)

    return (train_X,train_y,test_X,test_y)




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


if __name__=="__main__":
    (train_X, train_y, test_X, test_y)=LoadData()
    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)