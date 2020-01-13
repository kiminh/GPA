# -*- coding: utf-8 -*-
# @Time    : 2019/12/16 20:27
# @Author  : zxl
# @FileName: GpaVsFailed.py

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
import matplotlib.pyplot as plt

"""
挂科学生与未挂科学生gpa是否具有差异性
"""

if __name__ =="__main__":

    root="C://zxl/Data/GPA-large/processed/"
    gpa_file=root+"gpa.csv"
    df=pd.read_csv(gpa_file)

    for failed,group_df in df.groupby('failed'):
        print(failed)
        print("mean:%f"%group_df.gpa.mean())
        print("min:%f"%group_df.gpa.min())
        print("max:%f"%group_df.gpa.max())
        print("stu num:%d"%group_df.shape[0])
        plt.hist(group_df.gpa.values,color='blue',alpha=0.6)
        plt.title(str(failed))
        plt.show()
