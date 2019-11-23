# -*- coding: utf-8 -*-
# @Time    : 2019/11/23 11:36
# @Author  : zxl
# @FileName: test.py

import json
import numpy as np
import pandas as pd
import random
random.seed(10)
#
# df=pd.DataFrame({'a':[1,2,3,4],'b':[1,2,3,4],'c':[1,2,3,4]})
#
# df2=df.ix[:,:-2]
# df3=df.ix[:,-2:]
# print(df)
# print(df2)
# print(df3)

a=np.full(shape=(3,5),fill_value=3)
print(a)
print(a[:,1])

