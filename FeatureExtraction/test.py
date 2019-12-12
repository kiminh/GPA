# -*- coding: utf-8 -*-
# @Time    : 2019/11/23 11:36
# @Author  : zxl
# @FileName: test.py

import math
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
import matplotlib.pyplot as plt


arr=np.full(shape=(3,5),fill_value=3)

b=arr[:,0]+arr[:,1]+arr[:,2]
print(arr)
print(b)
