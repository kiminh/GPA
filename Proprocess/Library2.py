# -*- coding: utf-8 -*-
# @Time    : 2019/12/8 19:45
# @Author  : zxl
# @FileName: Library.py

import os
import numpy as np
import pandas as pd
from DB.DBUtil import DB
from Proprocess.Util import *
"""
学生学期前，学期中，学期末去图书馆次数
"""

def CalNum(semester,record_lst):
    num=[0,0,0]
    for str_date in record_lst:
        c=getSemInterval(semester,str_date)
        num[c]+=1
    return num



def SaveSta(library_dir,out_path):
    stu_lst = []
    fea_m = []

    for file_name in os.listdir(library_dir):
        stu_id = file_name[:-4]
        df = pd.read_csv(library_dir + file_name)

        for semester, group_df in df.groupby(['ctime']):
            semester=str(semester)
            records_set = set(group_df.time.values)
            stu_lst.append(stu_id)
            lib_num=CalNum(semester,records_set)
            fea_m.append([semester, lib_num[0],lib_num[1],lib_num[2]])
    fea_m = np.array(fea_m)
    total_lst=fea_m[:,1].astype(np.float32)+fea_m[:,2].astype(np.float32)+fea_m[:,3].astype(np.float32)
    df = pd.DataFrame({"stu_id": stu_lst, "ctime": fea_m[:,0],
                       "pre_lib_num": fea_m[:, 1],"dur_lib_num": fea_m[:, 2],"post_lib_num": fea_m[:, 3],
                       "lib_num":total_lst})
    df.to_csv(out_path, index=False)


if __name__ =="__main__":
    root = "C://zxl/Data/GPA-large/"
    profile_file=root+"stu/profile.csv"
    library_dir=root+"library/records/"
    new_library_dir=root+"new_library/records/"

    out_path=root+"processed/lib2.csv"


    #将学生每学期去图书馆次数存起来
    SaveSta(new_library_dir, out_path)


