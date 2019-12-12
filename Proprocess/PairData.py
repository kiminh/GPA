# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 16:03
# @Author  : zxl
# @FileName: PairData.py

import numpy as np
import pandas as pd

"""
将数据组织成(上一学期，当前学期)这样的pair
"""

def CalSemester(enroll_time,ctime):
    """
    根据入学时间和上课时间计算该学生修读当前课程的年级
    :param enroll_time:
    :param ctime:
    :return:
    """
    return (int(ctime[:4])-int(enroll_time))*2+int(ctime[4])-1

def PairData(consume_df,gpa_df,lib_df,profile_df,out_path):
    """
    将数据处理成pair，输出到文件
    :param consume_df:
    :param gpa_df:
    :param lib_df:
    :param profile_df:
    :param out_path:
    :return:
    """
    # consume_df=consume_df[consume_df['stu_id']=="D0C607BBDA6"]
    # gap_df=gpa_df[gpa_df['stu_id']=="D0C607BBDA6"]
    # lib_df=lib_df[lib_df['stu_id']=="D0C607BBDA6"]
    # profile_df=profile_df[profile_df['stu_id']=="D0C607BBDA6"]

    del_df = gpa_df[gpa_df['ctime'].str.contains('s')]
    gpa_df = gpa_df[gpa_df.ctime.isin(set(gpa_df.ctime).difference(set(del_df.ctime)))]
    profile_df['enroll_time']=[x[:4] for x in profile_df.enroll_time.values]
    df=pd.merge(lib_df,consume_df,on=['stu_id','ctime'])
    df=pd.merge(df,gpa_df,on=['stu_id','ctime'])
    df=pd.merge(df,profile_df,on=['stu_id'])#把个人信息也放进去了
    # del_df=df[df['ctime'].str.contains('s')]#删除暑期课程
    # df=df[df.ctime.isin(set(df.ctime).difference(set(del_df.ctime)))]

    df=df.sort_values(by="ctime",ascending=True)
    #把数据处理成(20151,20152)这样的格式，保存起来
    history_gpa=[]#上一学期gpa
    history_failed=[]#上一学期是否挂科
    history_credit=[]#上一学期学分
    history_grade=[]#上一学期年级
    cur_grade=[]#当前年级
    for stu_id,group_df in df.groupby("stu_id"):

        last_gpa=-1
        last_credit=-1
        last_failed=True
        for enroll_time,ctime,credit,gpa,failed in zip(group_df.enroll_time,group_df.ctime,group_df.credit,group_df.gpa,group_df.failed):
            history_gpa.append(last_gpa)
            history_credit.append(last_credit)
            history_failed.append(last_failed)

            last_gpa=gpa
            last_credit=credit
            last_failed=failed
            grade=CalSemester(enroll_time,ctime)
            history_grade.append(grade-1)
            cur_grade.append(grade)

    concat_df=pd.DataFrame({"history_gpa":history_gpa,"history_failed":history_failed,
                            "history_credit":history_credit,"history_grade":history_grade,"cur_grade":cur_grade})

    df=pd.concat([df,concat_df],axis=1)
    df=df[df["history_gpa"]!=-1]#第一个学期不构成pair，所以删除

    df.dropna(axis=0, how='any')#删去含有NAN的行
    df = pd.get_dummies(df, columns=["gender", "dep", "enroll_time"])#将信息按照one-hot形式表示
    df['failed']=[1-int(x) for x in df.failed.values]
    df['history_failed']=[1-int(x) for x in df.history_failed.values]
    df=df.drop(['ctime','birthday'],axis=1)

    df.to_csv(out_path,index=False)


if __name__=="__main__":
    root="C://zxl/Data/GPA-large/"
    consume_file=root+"processed/consume2.csv"
    gpa_file=root+"processed/gpa.csv"
    lib_file=root+"processed/lib2.csv"
    profile_file=root+"stu/profile.csv"
    out_path=root+"processed/pair2.csv"


    consume_df=pd.read_csv(consume_file,index_col=0)
    gpa_df=pd.read_csv(gpa_file)

    lib_df=pd.read_csv(lib_file)
    profile_df=pd.read_csv(profile_file)

    PairData(consume_df,gpa_df,lib_df,profile_df,out_path)
