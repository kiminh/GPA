# -*- coding: utf-8 -*-
# @Time    : 2019/12/10 16:58
# @Author  : zxl
# @FileName: FailedAnalylsis.py

import numpy as np
import pandas as pd

"""
看看那些绩点小于2的记录，是否挂科了
"""

if __name__ =="__main__":
    root="C://zxl/Data/GPA-large/"
    gpa_file=root+"processed/gpa.csv"

    low_gpa_file=root+"analysis/low_gpa.csv"
    failed_file=root+"analysis/failed.csv"

    thres=2
    gpa_df=pd.read_csv(gpa_file)
    low_gpa_df=gpa_df[gpa_df['gpa']<=thres]

    failed_df=gpa_df[gpa_df['failed']==True]

    # low_gpa_df.to_csv(low_gpa_file,index=False)
    # failed_df.to_csv(failed_file,index=False)
    #

    print("挂科学生总数:%d"%(failed_df.shape[0]))
    print("挂科学生中绩点小于%f人数为%d"%(thres,failed_df[failed_df['gpa']<=thres].shape[0]))

    print("绩点小于:%f的人数为%d"%(thres,low_gpa_df.shape[0]))
    print("绩点小于:%f中挂科人数为:%d"%(thres,low_gpa_df[low_gpa_df['failed']==True].shape[0]))






