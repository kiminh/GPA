# -*- coding: utf-8 -*-
# @Time    : 2019/12/7 15:53
# @Author  : zxl
# @FileName: Consume.py

import os
import re
import numpy as np
import pandas as pd
from DB.DBUtil import DB
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


def Save(stu_lst,save_root):
    for stu_id in stu_lst:

        # if enroll_time[:4]  not in ['2014','2015']:
        #     continue
        # if enroll_time[:4]=='2014':
        #     start_time="2016-09-04 00:00:00"
        #     end_time="2017-01-16 00:00:00"
        # else:
        #     start_time="2017-09-10 00:00:00"
        #     end_time="2018-01-22 00:00:00"
        start_time="2016-02-21 00:00:00"
        end_time="2018-07-09 00:00:00"
        recs=db.getConsumeRec(stu_id,start_time,end_time)
        if(len(recs)==0):
            continue
        recs=np.array(recs)
        rec_df=pd.DataFrame()
        rec_df['time']=recs[:,0]
        rec_df['place']=recs[:,1]
        rec_df['amount']=recs[:,2]
        rec_df['balence']=recs[:,3]
        rec_df.to_csv(save_root+stu_id+".csv",index=False)

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
        print(semester)
        seq = []
        break_count = 0  # 早餐次数
        lunch_count= 0
        dinner_count= 0
        for cur_date, group_df2 in group_df1.groupby(["date"]):
            time_lst = group_df2.time.values
            morning = True
            noon = True
            night = True

            for cur_time in time_lst:
                h = int(cur_time[:2])
                m = int(cur_time[3:5])
                if cur_time<="10:00:00":#早上
                    if morning:
                        morning=False
                        seq.append(bin(h,m))
                        break_count+=1
                elif cur_time<="16:00:00":#中午
                    if noon:
                        noon=False
                        seq.append(bin(h,m))
                        lunch_count+=1
                else :
                    if night:
                        night = False
                        seq.append(bin(h, m))
                        dinner_count += 1
        res.append([semester,break_count,lunch_count,dinner_count])
    return res

if __name__=="__main__":


    root="C://zxl/Data/GPA-large/"
    stu_file=root+"stu/stu_list.csv"
    save_dir=root+"consume/records/"
    complete_consume_dir=root+"new_consume/records/"
    statistic_path=root+"processed/consume.csv"
    db=DB()
    stu_df = pd.read_csv(stu_file)

    #从数据库取数据，存到文件
    # Save(stu_df.stu_id.values,save_dir)

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
    df=pd.DataFrame({"stu_id":stu_lst,"semester":fea_m[:,0],"breakfast_num":fea_m[:,1],"lunch_num":fea_m[:,2],"dinner_num":fea_m[:,3]})
    df.to_csv(statistic_path,index=False)



