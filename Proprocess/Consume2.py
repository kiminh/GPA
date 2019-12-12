# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 13:49
# @Author  : zxl
# @FileName: Consume2.py

import os
import numpy as np
import pandas as pd
from datetime import datetime

from DB.DBUtil import DB
from Proprocess.Util import *
"""
这个将消费数据再细化，学期前、学期中、学期后
"""



def bin(h,m):
    """
    把当前时间转为所在区间
    :param h: 小时，int类型
    :param m: 分钟，int类型
    :return: 区间
    """
    res=h*2
    if m>30:
        res+=2
    else:
        res+=1
    return res


def ExtractConsumeFea(df):
    """
    提取每位学生，每学期消费特征
    目前：早饭、午饭、晚饭、次数
    :param df:
    :return: [(semester,breakfast_num,lunch_num,dinner_num)]
    """
    df['date']=[x[:10] for x in df['time']]
    df['time']=[x[11:] for x in df['time']]
    res=[]
    for semester,group_df1 in df.groupby(['ctime']):
        # print(semester)
        semester=str(semester)
        break_count=[0,0,0]
        lunch_count=[0,0,0]
        dinner_count=[0,0,0]

        for cur_date, group_df2 in group_df1.groupby(["date"]):
            time_lst = group_df2.time.values
            morning = True
            noon = True
            night = True
            c=getSemInterval(semester,cur_date)#计算这是学期前，还是学期中，还是学期后

            for cur_time in time_lst:
                h = int(cur_time[:2])
                m = int(cur_time[3:5])
                if cur_time<="10:00:00":#早上
                    if morning:
                        morning=False
                        break_count[c]+=1
                elif cur_time<="16:00:00":#中午
                    if noon:
                        noon=False
                        lunch_count[c]+=1
                else :
                    if night:
                        night = False
                        dinner_count[c] += 1
        res.append([semester,break_count[0],lunch_count[0],dinner_count[0]
                       , break_count[1], lunch_count[1],dinner_count[1]
                       , break_count[2], lunch_count[2], dinner_count[2]])
    return res

if __name__=="__main__":


    root="C://zxl/Data/GPA-large/"
    stu_file=root+"stu/stu_list.csv"
    save_dir=root+"consume/records/"
    complete_consume_dir=root+"new_consume/records/"
    statistic_path=root+"processed/consume2.csv"
    db=DB()
    stu_df = pd.read_csv(stu_file)


    #统计学生每学期的消费特征
    stu_lst=[]
    fea_m=[]
    for file_name in os.listdir(complete_consume_dir):
        stu_id=file_name[:-4]
        df=pd.read_csv(complete_consume_dir+file_name)

        fea=ExtractConsumeFea(df)
        i=0
        while i<len(fea):
            stu_lst.append(stu_id)
            i+=1
        fea_m.extend(fea)

    fea_m=np.array(fea_m)
    df=pd.DataFrame({"stu_id":stu_lst,"semester":fea_m[:,0],"pre_breakfast_num":fea_m[:,1],"pre_lunch_num":fea_m[:,2],"pre_dinner_num":fea_m[:,3],
                     "dur_breakfast_num": fea_m[:, 4], "dur_lunch_num": fea_m[:, 5], "dur_dinner_num": fea_m[:, 6],
                     "post_breakfast_num": fea_m[:, 7], "post_lunch_num": fea_m[:, 8], "post_dinner_num": fea_m[:, 9],
                     "breakfast_num":fea_m[:,1].astype(np.float32)+fea_m[:,4].astype(np.float32)+fea_m[:,7].astype(np.float32),
                     "lunch_num": fea_m[:, 2].astype(np.float32) + fea_m[:, 5].astype(np.float32) + fea_m[:,8].astype(np.float32),
                     "dinner_num": fea_m[:, 3].astype(np.float32) + fea_m[:, 6].astype(np.float32) + fea_m[:,9].astype(np.float32),})
    df.to_csv(statistic_path,index=False)

