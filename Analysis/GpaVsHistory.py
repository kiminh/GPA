# -*- coding: utf-8 -*-
# @Time    : 2019/11/22 10:17
# @Author  : zxl
# @FileName: GpaVsHistory.py

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
import matplotlib.pyplot as plt

"""
学生历史成绩与当前学期成绩的关系
学生上一学期挂科与否是否会影响这一学期挂科与否

整体正相关，上学期绩点高，这学期也高
上学期挂科情况下，这学期挂科概率：0.512625
上学期通过情况下，这学期挂科概率: 0.068611

"""

if __name__ == "__main__":
    file_path="C://zxl/Data/GPA/grade/gpa.csv"
    gpa_df=pd.read_csv(file_path)
    hist_gpa_lst=[]
    cur_gpa_lst=[]
    hist_failed_lst=[]
    cur_failed_lst=[]


    for stu_id,stu_gpa_df2 in gpa_df.groupby('stu_id'):
        stu_gpa_df=stu_gpa_df2.sort_values(['ctime'],ascending=True)
        gpa_values=[]
        failed_values=[]
        for ctime,gpa,failed in zip(stu_gpa_df.ctime, stu_gpa_df.gpa, stu_gpa_df.failed):
            if ctime[-1]=='s':#不考虑暑假小学期
                continue
            gpa_values.append(gpa)
            failed_values.append(failed)
        for i in np.arange(1,len(gpa_values),1):
            cur_gpa_lst.append(gpa_values[i])
            cur_failed_lst.append(failed_values[i])
            hist_gpa_lst.append(gpa_values[i-1])
            hist_failed_lst.append(failed_values[i-1])


    #上一学期是否挂科与这学期挂科概率
    hist_failed_idx=np.argwhere(np.array(hist_failed_lst)==True).flatten()
    hist_pass_idx=np.argwhere(np.array(hist_failed_lst)==False).flatten()
    cur_failed_idx=np.argwhere(np.array(cur_failed_lst)==True).flatten()

    if len(hist_pass_idx)==0:
        failed_rate1=0
    else:
        failed_rate1=len(set(cur_failed_idx).intersection(set(hist_failed_idx)))/len(hist_failed_idx)
    if len(hist_pass_idx)==0:
        failed_rate2=0
    else:
        failed_rate2=len(set(hist_pass_idx).intersection(set(cur_failed_idx)))/len(hist_pass_idx)
    print("上学期挂科情况下，这学期挂科概率：%f"%failed_rate1)
    print("上学期通过情况下，这学期挂科概率: %f"%failed_rate2)


    plt.scatter(hist_gpa_lst,cur_gpa_lst)
    plt.show()




