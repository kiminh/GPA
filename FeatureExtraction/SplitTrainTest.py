# -*- coding: utf-8 -*-
# @Time    : 2019/11/23 10:31
# @Author  : zxl
# @FileName: SplitTrainTest.py

import os
import json
import random
random.seed(10)
import numpy as np
import pandas as pd

def CalSemester(enroll_time,ctime):
    """
    根据入学时间和上课时间计算该学生修读当前课程的年级
    :param enroll_time:
    :param ctime:
    :return:
    """
    return (int(ctime[:4])-int(enroll_time))*2+int(ctime[4])-1


def SplitStu(stu_df,grade_dir,entropy_df):
    stu_profile_dic = {}
    for stu_id, gender, enroll_time, dep in zip(stu_df.stu_id, stu_df.gender, stu_df.enroll_time, stu_df.dep):
        # if enroll_time[:4] not in ['2014', '2015']:#TODO
        #     continue
        stu_profile_dic[stu_id] = {}
        stu_profile_dic[stu_id]['gender'] = gender
        stu_profile_dic[stu_id]['enroll_time'] = enroll_time[:4]
        stu_profile_dic[stu_id]['dep'] = dep[2:-8]

    stu_course_dic = {}
    for file_name in os.listdir(grade_dir):
        stu_id = file_name[:-4]
        if stu_id not in stu_profile_dic.keys():  # 非14、15级学生
            continue
        enroll_time = stu_profile_dic[stu_id]['enroll_time']
        gpa_df = pd.read_csv(grade_dir + file_name)
        # 删去不合法记录
        gpa_df = gpa_df[gpa_df['cat'] != '']  # 无类别记录
        gpa_df = gpa_df[gpa_df['stu_num'] != 0]  # 无学生人数记录
        gpa_df = gpa_df[gpa_df['gpa'] != -1]  # gpa=-1的记录

        course_info = {}
        for course, credit, gpa, ctime, cat, stu_num in zip(gpa_df.course, gpa_df.credit, gpa_df.gpa, gpa_df.ctime,
                                                            gpa_df.cat, gpa_df.stu_num):
            ctime = str(ctime)
            if ctime[4] == 's':  # 删去暑假学期
                continue
            grade = CalSemester(enroll_time, ctime)
            if grade > 5:  # 预测第五个学期的，因此把这之后的都删了
                continue
            if grade not in course_info.keys():
                course_info[grade] = []
            course_info[grade].append([course, credit, gpa, ctime, cat, stu_num, gpa == 0])
        # print(stu_id+": %d学期课程记录"%len(course_info.keys()))
        if len(course_info.keys()) < 5:  # 保证这个学生每学期都有课
            continue
        stu_course_dic[stu_id] = course_info

    # 划分训练集和测试集，保证训练集测试集中dep一致，课程类别一致
    valid_stu_lst = list(set(stu_profile_dic.keys()).intersection(set(stu_course_dic.keys())))
    print("当前用户数目：%d" % len(valid_stu_lst))
    cur_train = set(random.sample(valid_stu_lst, int(len(valid_stu_lst) * 0.7)))
    cur_test = set(valid_stu_lst).difference(set(cur_train))
    print("当前train用户数目：%d" % len(cur_train))
    print("当前test用户数目：%d" % len(cur_test))
    changed = True
    while changed:
        changed = False

        invalid_stu_set = set([])

        # 保证trian和test中的dep是一致的
        train_dep = set([])
        for stu_id in cur_train:
            train_dep.add(stu_profile_dic[stu_id]['dep'])
        test_dep = set([])
        for stu_id in cur_test:
            test_dep.add(stu_profile_dic[stu_id]['dep'])

        valid_dep = set(set(train_dep).intersection(set(test_dep)))
        if len(valid_dep) != len(train_dep) or len(valid_dep) != len(test_dep):
            for stu_lst in [cur_train, cur_test]:
                for stu_id in stu_lst:
                    if stu_profile_dic[stu_id]['dep'] not in valid_dep:
                        invalid_stu_set.add(stu_id)

        # 保证trian和test中课程类别一致
        train_cat = set([])
        for stu_id in cur_train:  # 只在当前合法的用户空间内找
            sem_course = stu_course_dic[stu_id]
            for semester in sem_course.keys():
                for course in sem_course[semester]:
                    train_cat.add(course[4])
        test_cat = set([])
        for stu_id in cur_test:
            sem_course = stu_course_dic[stu_id]
            for semester in sem_course.keys():
                for course in sem_course[semester]:
                    test_cat.add(course[4])
        valid_cat = train_cat.intersection(test_cat)

        if len(valid_cat) != len(train_cat) or len(valid_cat) != len(test_cat):
            for stu_lst in [cur_train, cur_test]:
                for stu_id in stu_lst:
                    sem_course = stu_course_dic[stu_id]
                    for semester in sem_course.keys():
                        valid = False
                        for course in sem_course[semester]:
                            if course[4] in valid_cat:
                                valid = True
                                break
                        # 只要有一个semsester不合法，这个人都不合法
                        if not valid:
                            invalid_stu_set.add(stu_id)
                            break
                        cur_valid_courses = []
                        for course in sem_course[semester]:  # 更新每学期的课程
                            if course[4] in valid_cat:
                                cur_valid_courses.append(course)
                        stu_course_dic[stu_id][semester] = cur_valid_courses

        if len(invalid_stu_set) > 0:
            changed = True
        cur_train = cur_train.difference(invalid_stu_set)  # 更新训练集和测试集中的用户数目
        cur_test = cur_test.difference(invalid_stu_set)

        #保证train和test中用户有entropy记录
        cur_train=cur_train.intersection(set(entropy_df.stu_id.values))
        cur_test=cur_test.intersection(set(entropy_df.stu_id.values))

        print("当前train用户数目：%d" % len(cur_train))
        print("当前test用户数目：%d" % len(cur_test))

        train_stu_lst = list(cur_train)
        test_stu_lst = list(cur_test)
        return (train_stu_lst,test_stu_lst,stu_profile_dic,stu_course_dic)

def SplitProfile(stu_profile_dic,train_stu_lst,test_stu_lst,train_stu_path,test_stu_path):
    train_gender_lst = []
    test_gender_lst = []
    train_dep_lst = []
    test_dep_lst = []
    train_enroll_time_lst = []
    test_enroll_time_lst = []
    for stu_id in train_stu_lst:
        train_gender_lst.append(stu_profile_dic[stu_id]['gender'])
        train_dep_lst.append(stu_profile_dic[stu_id]['dep'])
        train_enroll_time_lst.append(stu_profile_dic[stu_id]['enroll_time'][:4])
    for stu_id in test_stu_lst:
        test_gender_lst.append(stu_profile_dic[stu_id]['gender'])
        test_dep_lst.append(stu_profile_dic[stu_id]['dep'])
        test_enroll_time_lst.append(stu_profile_dic[stu_id]['enroll_time'][:4])

    train_stu_df = pd.DataFrame({'stu_id': train_stu_lst, 'gender': train_gender_lst, 'dep': train_dep_lst,
                                 'enroll_time': train_enroll_time_lst})
    train_stu_df.to_csv(train_stu_path, index=None)
    test_stu_df = pd.DataFrame(
        {'stu_id': test_stu_lst, 'gender': test_gender_lst, 'dep': test_dep_lst, 'enroll_time': test_enroll_time_lst})
    test_stu_df.to_csv(test_stu_path, index=None)


def SplitCourse(stu_course_dic,train_stu_lst,test_stu_lst,train_gpa_json_path,test_gpa_json_path,train_gpa_csv_path,test_gpa_csv_path):
    train_stu_course_dic = {}
    test_stu_course_dic = {}
    for stu_id in train_stu_lst:
        train_stu_course_dic[stu_id] = stu_course_dic[stu_id]
    for stu_id in test_stu_lst:
        test_stu_course_dic[stu_id] = stu_course_dic[stu_id]

    jstr = json.dumps(train_stu_course_dic, indent=4, ensure_ascii=False)
    with open(train_gpa_json_path, 'w') as w:
        w.write(jstr)
    jstr = json.dumps(test_stu_course_dic, indent=4, ensure_ascii=False)
    with open(test_gpa_json_path, 'w') as w:
        w.write(jstr)

    train_gpa_m = []
    train_gpa_stu_lst = []
    train_sem_lst = []
    test_gpa_m = []
    test_gpa_stu_lst = []
    test_sem_lst = []
    for stu_id in train_stu_lst:
        for semester in stu_course_dic[stu_id]:
            for c in stu_course_dic[stu_id][semester]:
                train_gpa_m.append(c)
                train_gpa_stu_lst.append(stu_id)
                train_sem_lst.append(semester)
    for stu_id in train_stu_lst:
        for semester in stu_course_dic[stu_id]:
            for c in stu_course_dic[stu_id][semester]:
                test_gpa_m.append(c)
                test_gpa_stu_lst.append(stu_id)
                test_sem_lst.append(semester)
    # [course,credit,gpa,ctime,cat,stu_num,gpa==0]
    train_gpa_m = np.array(train_gpa_m)
    test_gpa_m = np.array(test_gpa_m)
    train_gpa_df = pd.DataFrame(
        {'stu_id': train_gpa_stu_lst, 'semester': train_sem_lst, 'cname': train_gpa_m[:, 0].flatten(),
         'credit': train_gpa_m[:, 1].flatten(), 'gpa': train_gpa_m[:, 2].flatten(), 'cat': train_gpa_m[:, 4].flatten(),
         'stu_num': train_gpa_m[:, 5].flatten(), 'failed': train_gpa_m[:, 6].flatten()})
    train_gpa_df.to_csv(train_gpa_csv_path, index=None)
    test_gpa_df = pd.DataFrame(
        {'stu_id': test_gpa_stu_lst, 'semester': test_sem_lst, 'cname': test_gpa_m[:, 0].flatten(),
         'credit': test_gpa_m[:, 1].flatten(), 'gpa': test_gpa_m[:, 2].flatten(), 'cat': test_gpa_m[:, 4].flatten(),
         'stu_num': test_gpa_m[:, 5].flatten(), 'failed': test_gpa_m[:, 6].flatten()})
    test_gpa_df.to_csv(test_gpa_csv_path, index=None)

def SplitEntropy(entropy_df,train_lst,test_lst,train_path,test_path):
    train_stu_df=pd.DataFrame({'stu_id':train_lst})
    test_stu_df=pd.DataFrame({'stu_id':test_lst})
    train_df=pd.merge(entropy_df,train_stu_df,on='stu_id')
    test_df=pd.merge(entropy_df,test_stu_df,on='stu_id')
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)




if __name__ =="__main__":
    root="C://zxl/Data/GPA/"
    profile_path=root+"stu/profile.csv"

    grade_dir=root+"grade/records_completed/"
    library_path=root+"entropy/library.csv"
    breakfast_path=root+"entropy/breakfast.csv"
    lunch_path=root+"entropy/lunch.csv"
    dinner_path=root+"entropy/dinner.csv"


    train_stu_path=root+"grade/train/train_stu.csv"
    test_stu_path=root+"grade/test/test_stu.csv"
    train_gpa_json_path=root+"train/gpa.json"
    test_gpa_json_path=root+"test/gpa.json"

    train_gpa_csv_path = root + "train/gpa.csv"
    test_gpa_csv_path = root + "test/gpa.csv"

    train_entropy_path=root+"train/entropy.csv"
    test_entropy_path=root+"test/entropy.csv"

    stu_df=pd.read_csv(profile_path)
    breakfast_df = pd.read_csv(breakfast_path)
    breakfast_df.columns=['stu_id','breakfast']
    lunch_df = pd.read_csv(lunch_path)
    lunch_df.columns=['stu_id','lunch']
    library_df=pd.read_csv(library_path)
    library_df.columns=['stu_id','library']

    dinner_df = pd.read_csv(dinner_path)
    entropy_df = pd.merge(breakfast_df, lunch_df, on="stu_id")
    entropy_df = pd.merge(entropy_df, dinner_df, on="stu_id")
    entropy_df=pd.merge(entropy_df,library_df,on="stu_id")


    train_stu_lst,test_stu_lst,stu_profile_dic,stu_course_dic=SplitStu(stu_df,grade_dir,entropy_df)
    #
    # SplitProfile(stu_profile_dic,train_stu_lst,test_stu_lst,train_stu_path,test_stu_path)
    #
    # SplitCourse(stu_course_dic,train_stu_lst,test_stu_lst,train_gpa_json_path,test_gpa_json_path,train_gpa_csv_path,train_gpa_csv_path)




    SplitEntropy(entropy_df,train_stu_lst,test_stu_lst,train_entropy_path,test_entropy_path)









