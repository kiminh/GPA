# -*- coding: utf-8 -*-
# @Time    : 2019/11/23 11:36
# @Author  : zxl
# @FileName: test.py

import json
import numpy as np
import pandas as pd
import random
random.seed(10)
np.random.seed(10)
from sklearn.metrics import f1_score,mean_squared_error,roc_auc_score

a=np.array([[1,2,3],[1,2,3]])

b=a[:,1]
b=np.reshape(b,newshape=(len(b),1))
c=a[:,0]
print(a)
print(b)
print(c)