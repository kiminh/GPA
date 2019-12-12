# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 14:42
# @Author  : zxl
# @FileName: GpaVsStatistic.py

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
import matplotlib.pyplot as plt

"""
基于processed中的三个文件，计算
GPA和消费，library的相关性
"""
def Draw(count_lst,gpa_lst,failed_lst,title):
    # fig,ax=plt.subplots(1,2)
    #
    # ax[0].scatter(count_lst,gpa_lst,color='blue')
    # # ax1.title('gpa')
    # ax[1].scatter(count_lst, failed_lst, color='red', marker='+')
    # # ax2.title('failed')
    # # plt.title(title)
    # fig.show()

    print(title)

    plt.scatter(count_lst, gpa_lst, color='blue')
    corr=round(pd.Series(count_lst).corr(pd.Series(gpa_lst)),4)
    plt.xlabel('count')
    plt.ylabel('avg_gpa')
    plt.title(title+": "+str(corr))
    print("gpa: %f"%corr)
    plt.show()
    plt.scatter(count_lst, failed_lst, color='red', marker='+')
    corr=round(pd.Series(count_lst).corr(pd.Series(failed_lst)),4)
    plt.xlabel('count')
    plt.ylabel('failed_rate')
    plt.title(title+": "+str(corr))
    print("failed_rate: %f"%corr)

    plt.show()

if __name__ =="__main__":
    root="C://zxl/Data/GPA-large/"
    processed_root=root+"processed/"
    consume_file=processed_root+"consume.csv"
    gpa_file=processed_root+"gpa.csv"
    lib_file=processed_root+"lib.csv"
    consume_file2=processed_root+"consume2.csv"
    lib_file2=processed_root+"lib2.csv"

    consume_df=pd.read_csv(consume_file)
    gpa_df=pd.read_csv(gpa_file)
    lib_df=pd.read_csv(lib_file)
    # consume_df2=pd.read_csv(consume_file2)
    # lib_df2=pd.read_csv(lib_file2)

    df=pd.merge(consume_df,gpa_df,on='stu_id')
    # df=pd.merge(df,lib_df,on='stu_id')
    # df=pd.merge(df,consume_df2,on='stu_id')
    df=pd.merge(df,lib_df,on='stu_id')

    for title in [
                  'breakfast_num','lunch_num','dinner_num','num',
                  # 'pre_breakfast_num','pre_lunch_num','pre_dinner_num',
                  # 'dur_breakfast_num', 'dur_lunch_num', 'dur_dinner_num',
                  # 'post_breakfast_num', 'post_lunch_num', 'post_dinner_num',
                  # 'pre_num','dur_num','post_num'
                  ]:
        count_lst=[]
        gpa_lst=[]
        failed_lst=[]
        for count, group_df in df.groupby(title):
            count_lst.append(count)
            gpa_lst.append(group_df['gpa'].mean())
            failed_lst.append(group_df['failed'].sum()/group_df.shape[0])
        Draw(count_lst,gpa_lst,failed_lst,title)





