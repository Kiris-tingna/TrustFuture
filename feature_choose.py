#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/18 9:08
# @Author  : Kiristingna
# @Site    : 
# @File    : feature_choose.py
# @Software: PyCharm

import pandas as pd
from pandas.tools.plotting import parallel_coordinates

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv("./data/train_feature.csv",  delim_whitespace=True, header=None)

features = StandardScaler().fit_transform(data.values[:, 0:-1])
# df = pd.DataFrame(features)
# plt.figure()
# bp = df.boxplot()
# plt.show()

# pca
pca = PCA(n_components=0.9)
pca.fit_transform(features)

print(pca.n_components_)
print(pca.explained_variance_ratio_)
