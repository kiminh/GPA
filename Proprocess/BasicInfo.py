# -*- coding: utf-8 -*-
# @Time    : 2019/7/15 16:26
# @Author  : zxl
# @FileName: BasicInfo.py

import numpy as np
import pandas as pd
from DB.DBUtil import DB

db = DB()

def getUnderGraByEnrollTime(enroll_time):
    res=db.getStuByEnrollTime(enroll_time)
    return res

def SaveDep(stu_list,dep_path):
    dep_lst = []
    for s in stu_list:
        dep_lst.append(db.getDepartmentByStuid(s))
    dep_df = pd.DataFrame({'stu_id': stu_list, 'dep': dep_lst})
    dep_df.to_csv(dep_path, index=False)

def SaveProf(stu_list,prof_path):
    prof_lst = []
    for stu_id in stu_list:
        prof_rec = db.getProfileById(stu_id)
        prof_lst.append(prof_rec)
    prof_lst = np.array(prof_lst)
    prof_df = pd.DataFrame({'stu_id': stu_list, 'gender': prof_lst[:, 0].flatten(), 'enroll_time':
        prof_lst[:,1].flatten(), 'birthday': prof_lst[:,2].flatten(), 'dep': prof_lst[:,3].flatten()})
    prof_df.to_csv(prof_path, index=False)


if __name__=="__main__":
    stu_list=[]
    for enroll_time in [2013,2014,2015,2016,2017]:
        cur_res=getUnderGraByEnrollTime(enroll_time)
        stu_list.extend(cur_res)
    file_path="C://zxl/Data/GPA-large/stu/stu_list.csv"
    #dep_path="C://zxl/Data/GPA-large/stu/dep.csv"
    profile_path = "C://zxl/Data/GPA-large/stu/profile.csv"
    stu_df=pd.DataFrame({'stu_id':stu_list})
    stu_df.to_csv(file_path,index=False)

    # SaveDep(stu_list,dep_path)

    SaveProf(stu_list,profile_path)


