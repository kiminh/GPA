# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 20:19
# @Author  : zxl
# @FileName: TaskCorr.py

import pandas as pd
import numpy as np

"""
看一下任务之间相关度
"""

if __name__ =="__main__":

    root="C://zxl/Data/GPA-large/processed/"
    gpa_file=root+"gpa.csv"
    gpa_df=pd.read_csv(gpa_file)

    tmp_df=gpa_df[['gpa','failed']]

    corr=tmp_df.corr()
    print(corr)