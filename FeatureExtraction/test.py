# -*- coding: utf-8 -*-
# @Time    : 2019/11/23 11:36
# @Author  : zxl
# @FileName: test.py

import math
import numpy as np
import pandas as pd

a=pd.DataFrame({'a':[1,2,3],'b':[5,6,7]})
b=pd.DataFrame({'a':[1,2,3],'c':[7,8,9]})

c=pd.merge(a,b,on='a')
print(a)
print(b)
print(c)