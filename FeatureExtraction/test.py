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
from sklearn.metrics import hamming_loss,precision_score,f1_score,accuracy_score,recall_score,coverage_error,label_ranking_loss,average_precision_score,roc_auc_score


arr=np.array([0.1,0.5,0.3,0.8])
idx=int(np.argmax(arr))
p=[0,0,0,0]
p[idx]=1
print(p)
