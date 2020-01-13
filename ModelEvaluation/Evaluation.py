# -*- coding: utf-8 -*-
# @Time    : 2020/1/13 9:34
# @Author  : zxl
# @FileName: Evaluation.py
from sklearn.metrics import roc_auc_score

def Evaluate(y1,y2):
    macro_auc = roc_auc_score(y1, y2, average="macro")
    print("macro auc:%f" % macro_auc)