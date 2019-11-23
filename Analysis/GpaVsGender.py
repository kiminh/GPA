# -*- coding: utf-8 -*-
# @Time    : 2019/11/19 21:24
# @Author  : zxl
# @FileName: GpaVsGender.py

import numpy as np
import pandas as pd

"""
不同性别学生gpa差异
女生挂科率低于男生，平均绩点明显高于男生
grade:男 
avg_gpa:2.983296 
failed_rate:0.535714 
grade:女 
avg_gpa:3.182167 
failed_rate:0.304410 
"""

if __name__ == "__main__":
    gpa_file="C://zxl/Data/GPA/grade/gpa.csv"
    stu_file="C://zxl/Data/GPA/stu/profile.csv"


    profile_df=pd.read_csv(stu_file)
    stu_gender_dic = {x:y for x,y in zip(profile_df.stu_id,profile_df.gender)}

    gender_gpa={}
    gpa_df=pd.read_csv(gpa_file)
    for stu_id,ctime,credit,gpa,failed in zip(gpa_df.stu_id,gpa_df.ctime,gpa_df.credit,gpa_df.gpa,gpa_df.failed):
        if credit==0:#未修读课程，忽略
            continue
        if stu_id not in stu_gender_dic.keys():
            continue

        gender=stu_gender_dic[stu_id]

        if gender not in gender_gpa.keys():
            gender_gpa[gender]={'credit':[],'gpa':[],'failed':[],'course_stu_num':[]}

        gender_gpa[gender]['course_stu_num'].append(stu_id)
        if failed:
            gender_gpa[gender]['failed'].append(stu_id)
        gender_gpa[gender]['credit'].append(credit)
        gender_gpa[gender]['gpa'].append(gpa)


    #不同年级的平均绩点差异、挂科率差异
    genders=['男','女']
    avg_gpa_lst=[]
    failed_lst=[]
    stu_lst=[]

    for g in genders:
        gender_gpa[g]['credit']=np.array(gender_gpa[g]['credit'])
        gender_gpa[g]['gpa']=np.array(gender_gpa[g]['gpa'])
        avg_gpa=np.sum(gender_gpa[g]['credit']*gender_gpa[g]['gpa'])/(np.sum(gender_gpa[g]['credit']))
        avg_gpa_lst.append(avg_gpa)
        course_stu_num=len((gender_gpa[g]['course_stu_num']))
        stu_lst.append(course_stu_num)
        failed_num=len((gender_gpa[g]['failed']))
        failed_lst.append(failed_num)
        print("grade:%s "%g)
        print("avg_gpa:%f "%avg_gpa)
        print("failed_rate:%f "%(failed_num/float(course_stu_num)))