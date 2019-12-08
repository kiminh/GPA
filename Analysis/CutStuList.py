# -*- coding: utf-8 -*-
# @Time    : 2019/12/8 19:54
# @Author  : zxl
# @FileName: CutStuList.py


import numpy as np
import pandas as pd


"""
分析对象为14,15级学生，所以把一些学生先删去
"""

if __name__=="__main__":
    root="C://zxl/Data/GPA/"
    stu_root=root+"stu/"
    profile_file=stu_root+"profile_old.csv"

    new_stu_file=stu_root+"stu_lst.csv"
    new_profile_file=stu_root+"profile.csv"

    profile_df=pd.read_csv(profile_file)

    stu_lst=[]
    gender_lst=[]
    enroll_time_lst=[]
    birthday_lst=[]
    dep_lst=[]

    for stu_id,gender,enroll_time,birthday,dep in zip(profile_df.stu_id,profile_df.gender,profile_df.enroll_time,profile_df.birthday,profile_df.dep):
        if enroll_time[:4] not in ['2014','2015']:
            continue

        stu_lst.append(stu_id)
        gender_lst.append(gender)
        enroll_time_lst.append(enroll_time)
        birthday_lst.append(birthday)
        dep_lst.append(dep)

    df=pd.DataFrame({'stu_id':stu_lst,'gender':gender_lst,'enroll_time':enroll_time_lst,'birthday':birthday_lst,'dep':dep_lst})
    df.to_csv(new_profile_file,index=False)

    df=pd.DataFrame({'stu_id':stu_lst})
    df.to_csv(new_stu_file,index=False)

