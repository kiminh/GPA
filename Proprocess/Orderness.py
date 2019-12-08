# -*- coding: utf-8 -*-
# @Time    : 2019/12/7 17:01
# @Author  : zxl
# @FileName: Orderness.py

import os
import math
import numpy as np
import pandas as pd
from Analysis.Util import *
"""
分析学生消费行为的规律性
后续还要分析其他行为
"""

def isSubSeq(s1,s2):
    """
    序列s2是否存在子序列等于s1
    :param s1: 子序列
    :param s2: 完整序列
    :return:
    """
    l1=len(s1)
    l2=len(s2)
    if l1>l2:
        return False
    for i in np.arange(0,l2-l1+1,1):
        if s2[i:i+l1]==s1:
            return True
    return False


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
def actual_entropy(seq):
    """
    论文中提到的真实熵
    :param seq:
    :return:
    """
    n=len(seq)
    if n==0:
        return -1
    lamda=[1]
    for i in np.arange(1,n,1):
        sub_len=0
        for j in np.arange(1,n-i+1,1):
            if not isSubSeq(seq[i:i+j],seq[0:i]):
                sub_len=j
                break
        lamda.append(sub_len)
    res=(1.0/((1.0/n)*np.sum(lamda)))*math.log(n)
    # print(lamda)
    return res
def info_entropy(lst):
    # 数据总个数
    num = len(lst)
    # 每个数据出现的次数
    numberofNoRepeat = dict()
    for data in lst:
        numberofNoRepeat[data] = numberofNoRepeat.get(data, 0) + 1
        # 打印各数据出现次数，以便核对
    print(numberofNoRepeat)
    # 返回信息熵，其中x/num为每个数据出现的频率
    return abs(sum(map(lambda x: x / num * math.log(x / num, 2), numberofNoRepeat.values())))




if __name__ =="__main__":
    root="C://zxl/Data/GPA/"
    consume_root=root+"consume/"
    entropy_path=root+"entropy/dinner.csv"

    stu_lst=[]
    entropy_lst=[]

    for file_name in os.listdir(consume_root):
        file_path=consume_root+file_name
        consume_df=pd.read_csv(file_path)

        date_lst=[]
        time_lst=[]
        for time,place in zip(consume_df.time,consume_df.place):
            if not isNormalRest(place):
                continue#暂时仅以在餐厅吃饭作为orderness的标准
            print(place)
            date_lst.append(time[:10])
            time_lst.append(time[-8:])

        df=pd.DataFrame({"date":date_lst,"time":time_lst})

        #开始处理这些序列
        seq=[]
        count = 0#早餐次数
        for cur_date, group_df in df.groupby(["date"]):
            time_lst=group_df.time.values
            morning=True
            noon=True
            night=True

            for cur_time in time_lst:
                h=int(cur_time[:2])
                m=int(cur_time[3:5])
                # if cur_time<="10:00:00":#早上
                #     if morning:
                #         morning=False
                #         seq.append(bin(h,m))
                #         count+=1
                # if cur_time<="16:00:00":#中午
                #     if noon:
                #         noon=False
                #         seq.append(bin(h,m))
                #         count+=1
                if cur_time>"16:00:00":
                    if night:
                        night=False
                        seq.append(bin(h,m))
                        count+=1
        # entropy=info_entropy(seq)
        entropy=count

        stu_lst.append(file_name[:-4])
        entropy_lst.append(entropy)

    df=pd.DataFrame({"stu_id":stu_lst,"entropy":entropy_lst})
    df.to_csv(entropy_path,index=False)



