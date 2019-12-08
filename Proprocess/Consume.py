# -*- coding: utf-8 -*-
# @Time    : 2019/12/7 15:53
# @Author  : zxl
# @FileName: Consume.py

import re
import numpy as np
import pandas as pd
from DB.DBUtil import DB


if __name__=="__main__":


    root="C://zxl/Data/GPA/"
    stu_file=root+"stu/profile.csv"
    save_dir=root+"consume/"
    db=DB()
    """
    将所有学生的消费记录存储到一个csv文件中
    """
    stu_df = pd.read_csv(stu_file)
    for stu_id,enroll_time in zip(stu_df.stu_id,stu_df.enroll_time):

        if enroll_time[:4]  not in ['2014','2015']:
            continue
        if enroll_time[:4]=='2014':
            start_time="2016-09-04 00:00:00"
            end_time="2017-01-16 00:00:00"
        else:
            start_time="2017-09-10 00:00:00"
            end_time="2018-01-22 00:00:00"
        recs=db.getConsumeRec(stu_id,start_time,end_time)
        if(len(recs)==0):
            continue
        recs=np.array(recs)
        rec_df=pd.DataFrame()
        rec_df['time']=recs[:,0]
        rec_df['place']=recs[:,1]
        rec_df['amount']=recs[:,2]
        rec_df['balence']=recs[:,3]
        rec_df.to_csv(save_dir+stu_id+".csv",index=False)