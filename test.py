# -*- coding: utf-8 -*-
# @Time    : 2019/11/20 10:17
# @Author  : zxl
# @FileName: test.py

import pandas as pd

df1=pd.DataFrame({"a":[1,2,3],"c":[1,2,3],"d":[1,2,3]})
df2=pd.DataFrame({"a":[1,2,3],"c":[1,2,3],"e":[1,2,3]})
df3=df1[['a','d']]
print(df1)
print(df2)
print(df3)