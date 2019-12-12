# -*- coding: utf-8 -*-
# @Time    : 2019/11/19 20:33
# @Author  : zxl
# @FileName: CourseInfo.py

import os
import numpy as np
import pandas as pd
import difflib

from DB.DBUtil import DB

"""
对原来的成绩文件进行扩展
增加：(课程类别，班级人数，教师) 三列 
"""
# 年级 、课程名称、院系 三者找，找不到就用年级和课程找
if __name__=="__main__":
    root="C://zxl/Data/GPA-large/"
    gpa_dir=root+"grade/records/"
    save_dir=root+"grade/records_completed/"
    profile_path=root+"stu/profile.csv"

    db=DB()

    processed_dic={}
    for file_name in os.listdir(save_dir):
        processed_dic[file_name[:-4]]=True

    profile_df=pd.read_csv(profile_path)
    stu_enrolltime={x:y[:4] for x,y in zip(profile_df.stu_id,profile_df.enroll_time)}
    stu_dep={x:y[2:-8] for x,y in zip(profile_df.stu_id,profile_df.dep)}
    for file_name in os.listdir(gpa_dir):
        stu_id=file_name[:-4]
        if stu_id in processed_dic.keys():
            continue
        enroll_time=stu_enrolltime[stu_id]
        dep=stu_dep[stu_id]
        gpa_df=pd.read_csv(gpa_dir+file_name)
        cat_lst=[]#课程类别
        stu_num_lst=[]#课程上课人数
        teacher_lst=[]#教师名称
        id_lst=[]#在数据库中对应的id
        sim_lst=[]
        for c_name in gpa_df.course.values:
            #找该课程可能的详细信息，最可能的匹配
            course_lst=db.SelectCourseInfo(enroll_time,c_name)

            max_sim=-1
            bm_type=''
            bm_num=0
            bm_teacher=''
            bm_id=-1
            for course_info in course_lst:#根据开课院系名称找最匹配的，如果都不匹配就随机
                course_dep=course_info[0]
                type=course_info[1]
                stu_num=course_info[2]
                teacher=course_info[3]
                id=course_info[4]

                sim_val=difflib.SequenceMatcher(None,dep,course_dep).quick_ratio()
                if sim_val > max_sim:
                    max_sim=sim_val
                    bm_type=type
                    bm_num=stu_num
                    bm_teacher=teacher
                    bm_id=id
            cat_lst.append(bm_type)
            stu_num_lst.append(bm_num)
            teacher_lst.append(bm_teacher)
            id_lst.append(bm_id)
            sim_lst.append(max_sim)

        info_df=pd.DataFrame({'cat':cat_lst,'stu_num':stu_num_lst,'teacher':teacher_lst,'id':id_lst,'sim':sim_lst})
        completed_gpa_df=pd.concat([gpa_df,info_df],axis=1)

        completed_gpa_df.to_csv(save_dir+file_name)















