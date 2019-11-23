# -*- coding: utf-8 -*-
# @Time    : 2019/11/19 21:45
# @Author  : zxl
# @FileName: GpaVsDep.py

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
import matplotlib.pyplot as plt

"""
不同院系学生gpa是否具有差异
gpa差异不明显
挂科率具有显著差异
"""


if __name__ == "__main__":
    gpa_file="C://zxl/Data/GPA/grade/gpa.csv"
    stu_file="C://zxl/Data/GPA/stu/profile.csv"


    profile_df=pd.read_csv(stu_file)
    stu_dep_dic = {x:y[2:-8] for x,y in zip(profile_df.stu_id,profile_df.dep)}

    dep_gpa={}
    gpa_df=pd.read_csv(gpa_file)
    for stu_id,ctime,credit,gpa,failed in zip(gpa_df.stu_id,gpa_df.ctime,gpa_df.credit,gpa_df.gpa,gpa_df.failed):
        if credit==0:#未修读课程，忽略
            continue
        if stu_id not in stu_dep_dic.keys():
            continue

        dep=stu_dep_dic[stu_id]

        if dep not in dep_gpa.keys():
            dep_gpa[dep]={'credit':[],'gpa':[],'failed':[],'course_stu_num':[]}

        dep_gpa[dep]['course_stu_num'].append(stu_id)
        if failed:
            dep_gpa[dep]['failed'].append(stu_id)
        dep_gpa[dep]['credit'].append(credit)
        dep_gpa[dep]['gpa'].append(gpa)


    #不同年级的平均绩点差异、挂科率差异
    deps=list(dep_gpa.keys())
    avg_gpa_lst=[]
    failed_lst=[]
    stu_lst=[]

    for g in deps:
        dep_gpa[g]['credit']=np.array(dep_gpa[g]['credit'])
        dep_gpa[g]['gpa']=np.array(dep_gpa[g]['gpa'])
        avg_gpa=np.sum(dep_gpa[g]['credit']*dep_gpa[g]['gpa'])/(np.sum(dep_gpa[g]['credit']))
        avg_gpa_lst.append(avg_gpa)
        course_stu_num=len((dep_gpa[g]['course_stu_num']))
        stu_lst.append(course_stu_num)
        failed_num=len((dep_gpa[g]['failed']))
        failed_lst.append(failed_num/float(course_stu_num))
        print("grade:%s "%g)
        print("avg_gpa:%f "%avg_gpa)
        print("failed_rate:%f "%(failed_num/float(course_stu_num)))


    plt.bar(np.arange(0,len(deps)),avg_gpa_lst)
    plt.show()

    plt.bar(np.arange(0,len(deps)),failed_lst)
    plt.show()