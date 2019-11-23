# -*- coding: utf-8 -*-
# @Time    : 2019/11/22 9:21
# @Author  : zxl
# @FileName: GpaVsStuNum.py

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
import matplotlib.pyplot as plt

"""

课程人数，课程学分
与gpa和挂科率的关系

明显看出：班级人数越多，挂科率越低（与前面看到的通识课挂科率低相符）
"""
if __name__ == "__main__":
    gpa_dir="C://zxl/Data/GPA/grade/records_completed/"

    cid_records={}

    for file_name in os.listdir(gpa_dir):
        gpa_df=pd.read_csv(gpa_dir+file_name)
        for id,credit,stu_num,gpa in zip(gpa_df.id,gpa_df.credit,gpa_df.stu_num,gpa_df.gpa):
            if gpa==-1:#暂时不考虑这种无gpa的课程
                continue
            if credit>8 :
                print("credit:%d"%credit)
                continue
            if id not in cid_records.keys():
                cid_records[id]={}
                cid_records[id]['stu_num']=0
                cid_records[id]['gpa_lst']=[]
                cid_records[id]['failed_num']=0
                cid_records[id]['credit']=credit
            cid_records[id]['stu_num']+=1
            if gpa==0:
                cid_records[id]['failed_num']+=1
            else:
                cid_records[id]['gpa_lst'].append(gpa)

    credit_lst=[]
    stu_num_lst=[]
    avg_gpa_lst=[]
    failed_rate_lst=[]
    for cid in cid_records.keys():
        credit_lst.append(cid_records[cid]['credit'])
        stu_num_lst.append(cid_records[cid]['stu_num'])
        if len(cid_records[cid]['gpa_lst'])==0:
            avg_gpa_lst.append(0)
            failed_rate_lst.append(1)
        else:
            avg_gpa_lst.append(np.mean(cid_records[cid]['gpa_lst']))
            failed_rate_lst.append(cid_records[cid]['failed_num']/cid_records[cid]['stu_num'])
            print("failed_num:%f"%cid_records[cid]['failed_num'])
            print("stu_num:%f"%cid_records[cid]['stu_num'])

    #学分VS成绩
    plt.scatter(credit_lst,avg_gpa_lst)
    plt.show()

    plt.scatter(credit_lst,failed_rate_lst)
    plt.show()

    #学生人数VS成绩
    plt.scatter(stu_num_lst,avg_gpa_lst)
    plt.show()

    plt.scatter(stu_num_lst,failed_rate_lst)
    plt.show()






