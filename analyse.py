#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/23 15:00
# @Author  : Kiristingna
# @Site    : 
# @File    : analyse.py
# @Software: PyCharm

import pandas as pd
import numpy as np

dataframe_1= pd.read_csv("dsjtzs_txfzjh_preliminary.txt", delim_whitespace=True, header=None,
                            names=['id'])  # load dataset
dataframe_2 = pd.read_csv("wqe.txt", delim_whitespace=True, header=None,
                            names=['id'])  # load dataset

d1 = dataframe_1['id'].values
d2 = dataframe_2['id'].values
same = np.intersect1d(d1, d2)
# print(same)
# np.savetxt('result2.csv', same, fmt="%s")

TN = len(same) / 20000
TP = 1
# print(len(same), len(same) / min(len(d1), len(d2)))
print(5*TP*TN / (2*TP + 3 * TN))