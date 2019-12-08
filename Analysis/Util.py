# -*- coding: utf-8 -*-
# @Time    : 2019/12/7 17:04
# @Author  : zxl
# @FileName: Util.py

import re

"""
一些需要用到的函数
"""
#是否是正常餐饮
def isNormalRest(location):
    pattern = re.compile(r"(.*餐厅.*)|(.*餐车.*)|(^秋实阁)|(^闵行秋林阁)|(^中江路秋林阁)")
    arr = pattern.findall(location)
    return len(arr) > 0
#是否是零食消费
def isSnacks(location):
    pattern = re.compile(r"(^华夏商场)|(.*便利店.*)|(.*超市.*)|"
                         r"(.*面包房.*)|(.*咖啡.*)|(.*牛奶棚$)|(.*小卖部.*)")
    arr = pattern.findall(location)
    return len(arr) > 0
#是否是淋浴
def isShower(location):
    pattern = re.compile(r"(.*浴室.*)|(^闵行本科生公寓)|(^闵行校区研究生公寓)")
    arr = pattern.findall(location)
    return len(arr) > 0
#是否是取款
def isReceive(location):
    pattern=re.compile(r"(^中北领款工作站)|(^中北出纳机)|(^圈存机)|(^自主补卡前置服务)|(.*领款.*)|(.*圈存.*)|(.*出纳.*)")
    arr=pattern.findall(location)
    return len(arr)>0

