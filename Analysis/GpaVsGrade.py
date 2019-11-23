# -*- coding: utf-8 -*-
# @Time    : 2019/11/19 20:39
# @Author  : zxl
# @FileName: GpaVsGrade.py

import numpy as np
import pandas as pd

"""
不同年级学生gpa的差异性
"""
"""
不同年级学生gpa没有明显差异，挂科率呈现下降趋势
"""

if __name__ == "__main__":
    gpa_file="C://zxl/Data/GPA/grade/gpa.csv"
    stu_file="C://zxl/Data/GPA/stu/profile.csv"


    profile_df=pd.read_csv(stu_file)
    stu_enroll_dic = {x:int(y[:4]) for x,y in zip(profile_df.stu_id,profile_df.enroll_time)}

    grade_gpa={}
    gpa_df=pd.read_csv(gpa_file)
    for stu_id,ctime,credit,gpa,failed in zip(gpa_df.stu_id,gpa_df.ctime,gpa_df.credit,gpa_df.gpa,gpa_df.failed):
        if credit==0:#未修读课程，忽略
            continue
        sem=ctime[4]
        cur_ctime=int(ctime[:4])
        if sem=='2':
            cur_ctime+=1
        grade=cur_ctime-stu_enroll_dic[stu_id]#这个同学的年级

        if grade not in grade_gpa.keys():
            grade_gpa[grade]={'credit':[],'gpa':[],'failed':[],'course_stu_num':[]}

        grade_gpa[grade]['course_stu_num'].append(stu_id)
        if failed:
            grade_gpa[grade]['failed'].append(stu_id)
        grade_gpa[grade]['credit'].append(credit)
        grade_gpa[grade]['gpa'].append(gpa)


    #不同年级的平均绩点差异、挂科率差异
    grades=[1,2,3,4]
    avg_gpa_lst=[]
    failed_lst=[]
    stu_lst=[]

    for g in grades:
        grade_gpa[g]['credit']=np.array(grade_gpa[g]['credit'])
        grade_gpa[g]['gpa']=np.array(grade_gpa[g]['gpa'])
        avg_gpa=np.sum(grade_gpa[g]['credit']*grade_gpa[g]['gpa'])/(np.sum(grade_gpa[g]['credit']))
        avg_gpa_lst.append(avg_gpa)
        course_stu_num=len((grade_gpa[g]['course_stu_num']))
        stu_lst.append(course_stu_num)
        failed_num=len((grade_gpa[g]['failed']))
        failed_lst.append(failed_num)
        print("grade:%d"%g)
        print("avg_gpa:%f"%avg_gpa)
        print("failed_rate:%f"%(failed_num/float(course_stu_num)))











