# -*- coding: utf-8 -*-
# @Time    : 2019/12/7 20:33
# @Author  : zxl
# @FileName: GpaVsEntropy.py

from sklearn import preprocessing

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
import matplotlib.pyplot as plt


"""
分析gpa、挂科率与orderness关系
"""

if __name__ == "__main__":
    root="C://zxl/Data/GPA/"
    gpa_file=root+"grade/gpa.csv"
    stu_file=root+"stu/profile.csv"
    entropy_file=root+"entropy/lunch.csv"

    stu_df=pd.read_csv(stu_file)
    gpa_df=pd.read_csv(gpa_file)
    entropy_df=pd.read_csv(entropy_file)



    stu_enroll_dic={}
    stu_gpa_dic={}
    stu_failed_dic={}
    for stu_id,enroll_time in zip(stu_df.stu_id,stu_df.enroll_time):
        if enroll_time[:4] not in ['2014','2015']:
            continue
        stu_enroll_dic[stu_id]=enroll_time[:4]
    for stu_id,ctime,gpa,failed in zip(gpa_df.stu_id,gpa_df.ctime,gpa_df.gpa,gpa_df.failed):
        if stu_id not in stu_enroll_dic.keys():
            continue
        enroll_time=stu_enroll_dic[stu_id]
        if (enroll_time=='2014' and ctime =='20162') or (enroll_time=='2015' and ctime=='20172'):
            stu_gpa_dic[stu_id]=gpa
            stu_failed_dic[stu_id]=failed

    stu_lst = []
    gpa_lst = []
    failed_lst = []
    for stu_id in set(stu_enroll_dic.keys()).intersection(stu_gpa_dic.keys()):
        stu_lst.append(stu_id)
        gpa_lst.append(stu_gpa_dic[stu_id])
        failed_lst.append(stu_failed_dic[stu_id])

    gpa_df=pd.DataFrame({"stu_id":stu_lst,"gpa":gpa_lst,"failed":failed_lst})

    df=pd.merge(entropy_df,gpa_df,on="stu_id")
    df=df[df['entropy']!=-1]
    # df = df[df['entropy'] != 0]
    # df = df[df['gpa'] != 0]

    entropy_lst=df.entropy.values
    gpa_lst=df.gpa.values
    print("stu number:%d"%(len(entropy_lst)))
    # entropy_lst=preprocessing.scale(entropy_lst)
    # gpa_lst=preprocessing.scale(gpa_lst)

    plt.scatter(entropy_lst,gpa_lst)
    plt.show()







