# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 14:58
# @Author  : zxl
# @FileName: Util.py

from datetime import datetime

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

sem_dic = {}
s_d1 = '2016-02-21'
s_d2 = '2016-07-04'
sem_dic['20161'] = {}
sem_dic['20161']['time'] = [s_d1, s_d2]
sem_dic['20161']['days'] = (datetime.strptime(s_d2, '%Y-%m-%d') - datetime.strptime(s_d1, '%Y-%m-%d')).days
s_d1 = '2016-09-04'
s_d2 = '2017-01-16'
sem_dic['20162'] = {}
sem_dic['20162']['time'] = [s_d1, s_d2]
sem_dic['20162']['days'] = (datetime.strptime(s_d2, '%Y-%m-%d') - datetime.strptime(s_d1, '%Y-%m-%d')).days
s_d1 = '2017-02-14'
s_d2 = '2017-07-03'
sem_dic['20171'] = {}
sem_dic['20171']['time'] = [s_d1, s_d2]
sem_dic['20171']['days'] = (datetime.strptime(s_d2, '%Y-%m-%d') - datetime.strptime(s_d1, '%Y-%m-%d')).days
s_d1 = '2017-09-10'
s_d2 = '2018-01-22'
sem_dic['20172'] = {}
sem_dic['20172']['time'] = [s_d1, s_d2]
sem_dic['20172']['days'] = (datetime.strptime(s_d2, '%Y-%m-%d') - datetime.strptime(s_d1, '%Y-%m-%d')).days
s_d1 = '2018-02-25'
s_d2 = '2018-07-09'
sem_dic['20181'] = {}
sem_dic['20181']['time'] = [s_d1, s_d2]
sem_dic['20181']['days'] = (datetime.strptime(s_d2, '%Y-%m-%d') - datetime.strptime(s_d1, '%Y-%m-%d')).days

def getSemInterval(semester,str_date):
    """
    根据semester，输出当前日期是学期前，还是学期中，还是学期后
    对于s与w，都是前
    0：学期前
    1：学期中
    2：学期后
    :param semester: 20161，str
    :param date: 2016-03-21
    :return: 0
    """
    if semester[4] in ['s', 'w']:
        return 0
    cur_d1=sem_dic[semester]['time'][0]
    total_day=sem_dic[semester]['days']
    intervals=(datetime.strptime(str_date, '%Y-%m-%d')-datetime.strptime(cur_d1, '%Y-%m-%d')).days
    if intervals/total_day <1.0/3:
        return 0
    if intervals/total_day<2.0/3:
        return 1
    return 2
