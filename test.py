# -*- coding: utf-8 -*-
# @Time    : 2019/11/20 10:17
# @Author  : zxl
# @FileName: test.py

import pandas as pd
import numpy as np


arr=np.array([1,2,3,4])
b=np.reshape(arr,newshape=(-1,1))
print(arr)
print(b)