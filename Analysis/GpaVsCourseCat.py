# -*- coding: utf-8 -*-
# @Time    : 2019/11/19 21:44
# @Author  : zxl
# @FileName: GpaVsCourseCat.py

import os
import numpy as np
import pandas as pd
"""
不同课程类别绩点有差异，但是不明显
不同课程类别的挂科率具有一定差异（数学基础课程挂科率较高）
"""
if __name__ == "__main__":
    gpa_dir="C://zxl/Data/GPA/grade/records_completed/"

    cat_records={}
    for file_name in os.listdir(gpa_dir):
        gpa_df=pd.read_csv(gpa_dir+file_name)
        for cat,credit,gpa in zip(gpa_df.cat,gpa_df.credit,gpa_df.gpa):
            if gpa==-1:#暂时不考虑这种无gpa的课程
                continue
            if cat not in cat_records.keys():
                cat_records[cat]={}
                cat_records[cat]['total_credit']=0.0
                cat_records[cat]['total_gpa']=0.0
                cat_records[cat]['record_num']=0
                cat_records[cat]['failed_num']=0
            cat_records[cat]['total_credit']+=float(credit)
            cat_records[cat]['total_gpa']+=float(credit)*float(gpa)
            cat_records[cat]['record_num']+=1
            if gpa==0:#挂科了
                cat_records[cat]['failed_num']+=1


    avg_gpa_lst=[]
    failed_rate_lst=[]
    for cat in cat_records.keys():
        avg_gpa=cat_records[cat]['total_gpa']/cat_records[cat]['total_credit']
        failed_rate=cat_records[cat]['failed_num']/cat_records[cat]['record_num']
        avg_gpa_lst.append(avg_gpa)
        failed_rate_lst.append(failed_rate)
        print(cat)
        print("avg_gpa:%f "%avg_gpa)
        print("failed_rate:%f"%failed_rate)



