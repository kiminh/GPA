# -*- coding: utf-8 -*-
# @Time    : 2019/9/30 12:56
# @Author  : zxl
# @FileName: Grade.py

import os
import pandas as pd
from DB.DBUtil import DB
db=DB()
"""
这个文件把stu_id, dep ,course,credit,grade ,gpa存到文件里
"""


"""
semester_list=[20162,20171,2017s]
"""
def getGrade(stu_id,semester_list):
    course_list=[]
    credit_list=[]
    grade_list=[]
    level_list=[]
    gpa_list=[]
    ctime_list=[]
    if semester_list is  None:
        cur_res = db.getAllGradeInfo(stu_id)
        courses = [x[0] for x in cur_res]
        credits = [x[1] for x in cur_res]
        grades = [x[2] for x in cur_res]
        levels = [x[3] for x in cur_res]
        gpas = [x[4] for x in cur_res]
        ctimes=[x[5] for x in cur_res]

        course_list.extend(courses)
        credit_list.extend(credits)
        grade_list.extend(grades)
        level_list.extend(levels)
        gpa_list.extend(gpas)
        ctime_list.extend(ctimes)
    else:
        for semester in semester_list:

            cur_res=db.getGradeInfo(stu_id,semester)
            courses=[x[0] for x in cur_res]
            credits=[x[1] for x in cur_res]
            grades=[x[2] for x in cur_res]
            levels=[x[3] for x in cur_res]
            gpas=[x[4] for x in cur_res]
            ctimes=[x[5] for x in cur_res]

            course_list.extend(courses)
            credit_list.extend(credits)
            grade_list.extend(grades)
            level_list.extend(levels)
            gpa_list.extend(gpas)
            ctime_list.extend(ctimes)

    res={}
    res['course']=course_list
    res['credit']=credit_list
    res['grade']=grade_list
    res['level']=level_list
    res['gpa']=gpa_list
    res['ctime']=ctime_list
    return res


"""
输入：这个人这一学年各科成绩
返回：这个人这一学年平均绩点
"""
def CalGpa(gpa_df):
    """
    学生绩点为-1：这门课只是按照通过与否计算，不算入绩点
    绩点为0：挂科
    绩点为1：补考通过，能够参评奖学金
    :param gpa_df:
    :return:
    """
    failed=False
    total_credit=0
    sum_gpa=0
    for credit,grade,level,gpa in zip(gpa_df.credit,gpa_df.grade,gpa_df.level,gpa_df.gpa):
        if gpa == -1:
            continue
        elif gpa ==0 :
            total_credit+=credit
            failed=True
        else:
            total_credit+=credit
            sum_gpa+=(gpa*credit)

    avg_gpa=0
    if total_credit!=0:#如果total_credit=0，说明没有（正常的）课，平均绩点就是0
        avg_gpa=sum_gpa/total_credit
    return (total_credit,avg_gpa,failed)


if __name__ =="__main__":
    root="C://zxl/Data/GPA-large/"
    stu_file=root+"stu/stu_list.csv"
    save_root=root+"grade/"
    save_dir=save_root+"records/"
    # semesters=['20162','20171','2017s']
    stu_df= pd.read_csv(stu_file)

    processed_dic={}
    for file_name in os.listdir(save_dir):
        processed_dic[file_name[:-4]]=True


    """
    从数据库里面找成绩记录，将每个人的成绩记录存到一个csv文件中
    """
    # for stu_id in stu_df.stu_id.values:
    #     if stu_id in processed_dic.keys():#上次已经找过的记录，这次不再查找
    #         continue
    #     dic=getGrade(stu_id,None)
    #     df=pd.DataFrame(dic)
    #     save_path=save_dir+stu_id+".csv"
    #     df.to_csv(save_path,index=False)
    #     print(stu_id)

    """
    基于每个人的成绩记录，
    计算每个学生每学期的平均绩点
    """
    stu_list=[]
    ctime_list=[]
    gpa_list=[]
    failed_list=[]
    total_credit_list=[]
    for stu_id in stu_df.stu_id.values[:10]:
        gpa_df=pd.read_csv(save_dir+stu_id+".csv")
        for ctime,df in gpa_df.groupby('ctime'):
            (total_credit,avg_gpa,failed)=CalGpa(df)
            if total_credit==0:#如果这个学生这学期没有课，删除这条记录
                continue
            stu_list.append(stu_id)
            ctime_list.append(ctime)
            gpa_list.append(avg_gpa)
            failed_list.append(failed)
            total_credit_list.append(total_credit)

    dic={}
    dic['stu_id']=stu_list
    dic['ctime']=ctime_list
    dic['credit']=total_credit_list
    dic['gpa']=gpa_list
    dic['failed']=failed_list

    df=pd.DataFrame(dic)
    df.to_csv(root+"processed/gpa.csv",index=False)