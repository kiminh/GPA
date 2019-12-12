# -*- coding: utf-8 -*-
# @Time    : 2019/12/9 14:13
# @Author  : zxl
# @FileName: AddColumn.py

import os
import numpy as np
import pandas as pd

"""
按照consume和library时间，给每个文件都加一列semester
"""

def CalSemester(cur_date):
    """
    对于给定日期，输出该日期是哪个学期
    暑假：2014s
    寒假: 2014w
    :param cur_date: 2014-02-21
    :return: 20141
    """
    """
    学校校历：
    2016-02-21至2016-07-04: 20161
    2016-07-04至2016-09-04：2016s
    2016-09-04至2017-01-16: 20162
    2017-01-16至2017-02-14: 2017w
    2017-02-14至2017-07-03: 20171
    2017-07-03至2017-09-10: 2017s
    2017-09-10至2018-01-22: 20172
    2018-01-22至2018-02-25: 2018w
    2018-02-25至2018-07-09: 20181
    """

    cur_date=str(cur_date)[:10]
    if cur_date>="2016-02-21" and cur_date<="2016-07-04":
        return "20161"
    elif cur_date>="2016-07-04" and cur_date<="2016-09-04":
        return "2016s"
    elif cur_date>="2016-09-04" and cur_date<="2017-01-16":
        return "20162"
    elif cur_date>="2017-01-16" and cur_date<="2017-02-14":
        return "2017w"
    elif cur_date>="2017-02-14" and cur_date<="2017-07-03":
        return "20171"
    elif cur_date>="2017-07-03" and cur_date<="2017-09-10":
        return "2017s"
    elif cur_date>="2017-09-10" and cur_date<="2018-01-22":
        return "20172"
    elif cur_date>="2018-01-22" and cur_date<="2018-02-25":
        return "2018w"
    elif cur_date>="2018-02-25" and cur_date<="2018-07-09":
        return "20181"
    else:
        print("该日期不再范围内"+cur_date)
        return "None"


if __name__ =="__main__":
    root="C://zxl/Data/GPA-large/"
    library_dir=root+"library/records/"
    consume_dir=root+"consume/"

    new_library_dir=root+"new_library/records/"
    new_consume_dir=root+"new_consume/records/"

    for file_name in os.listdir(library_dir):
        df=pd.read_csv(library_dir+file_name)
        sem_lst=[]
        for t in df.time.values:
            sem=CalSemester(t)
            sem_lst.append(sem)
        sem_df=pd.DataFrame({"ctime":sem_lst})
        df=pd.concat([df,sem_df],axis=1)

        df.to_csv(new_library_dir+file_name,index=False)

    # for file_name in os.listdir(consume_dir):
    #     df=pd.read_csv(consume_dir+file_name)
    #     sem_lst=[]
    #     for t in df.time.values:
    #         sem=CalSemester(t)
    #         sem_lst.append(sem)
    #     sem_df=pd.DataFrame({"ctime":sem_lst})
    #     df=pd.concat([df,sem_df],axis=1)
    #
    #     df.to_csv(new_consume_dir+file_name,index=False)





