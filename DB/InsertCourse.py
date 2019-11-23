# -*- coding: utf-8 -*-
# @Time    : 2019/11/19 15:29
# @Author  : zxl
# @FileName: InsertCourse.py

import pandas as pd
from DB.DBUtil import DB

if __name__ =="__main__":
    db=DB()
    file_path="C://zxl/Data/StudyRelated/course.csv"
    df=pd.read_csv(file_path)
    df.columns=['idx1','dep_id','dep_name','c_id','c_name','stu_num','grade','type','teacher','week','week_num','section','classroom','campus']
    i=1
    for(dep_id,dep_name,c_id,c_name,stu_num,grade,type,teacher,week,week_num,section,classroom,campus) in zip(df.dep_id,df.dep_name,df.c_id,df.c_name,df.stu_num,df.grade,df.type,df.teacher,df.week,df.week_num,df.section,df.classroom,df.campus):
        c_name=str(c_name)
        teacher=str(teacher)
        c_name=c_name.replace('\'','')
        teacher=teacher.replace('\'','')
        if str(week_num)=='nan':
            week_num='0'
        print(i)
        i+=1

        # dep_name=dep_name.replace('\n','')
        # tmp=dep_name+'bbb'
        # print(tmp)

        db.InsertCourse(dep_id,dep_name,c_id,c_name,stu_num,grade,type,teacher,week,week_num,section,classroom,campus)

