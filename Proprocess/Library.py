# -*- coding: utf-8 -*-
# @Time    : 2019/12/8 19:45
# @Author  : zxl
# @FileName: Library.py

import os
import numpy as np
import pandas as pd
from DB.DBUtil import DB

"""
找学生去图书馆次数
  工作日次数
  周末次数
  寒暑假次数
因为只有某一天，所以只能记录是否去过图书馆
"""


def Save(profile_file,library_dir):
    profile_df = pd.read_csv(profile_file)

    for stu_id, enroll_time in zip(profile_df.stu_id, profile_df.enroll_time):
        if enroll_time[:4] == '2014':
            start_time = "2016-09-04 00:00:00.000"
            end_time = "2017-01-16 00:00:00.000"
        else:
            start_time = "2017-09-10 00:00:00.000"
            end_time = "2018-01-22 00:00:00.000"
        lib_records = db.getLibInfo(stu_id, start_time, end_time)

        out_path = library_dir + stu_id + ".csv"
        df = pd.DataFrame({"time": lib_records})
        df.to_csv(out_path, index=False)


if __name__ =="__main__":
    root = "C://zxl/Data/GPA/"
    profile_file=root+"stu/profile.csv"
    library_dir=root+"library/"

    out_path=root+"entropy/library.csv"

    db=DB()

    # Save(profile_file,library_dir)#将去图书馆的记录存起来


    stu_lst=[]
    num_lst=[]
    for file_name in os.listdir(library_dir):
        stu_id=file_name[:-4]
        df=pd.read_csv(library_dir+file_name)
        records_set=set(df.time.values)
        stu_lst.append(stu_id)
        num_lst.append(len(records_set))

    df=pd.DataFrame({"stu_id":stu_lst,"num":num_lst})
    df.to_csv(out_path,index=False)

