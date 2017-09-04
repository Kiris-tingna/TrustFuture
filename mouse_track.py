#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/5/21 18:05
# @Author  : Kiristingna
# @File    : mouse_track.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from feature_extraction import feature_selection
np.seterr(all='ignore')

# ----------------- sampling --------------#
# dataframe_train = pd.read_csv("data/dsjtzs_txfz_training.txt", delim_whitespace=True, header=None,
#                             names=['id', 'sequence', 'target', 'cate'])  # load dataset
# X, Y, del_ids = feature_selection(dataframe_train, 'train')

# --------------- pre train --------------#
feature = pd.read_csv('./data/train_feature.csv', delim_whitespace=True, header=None)
X = feature.values[:, 0:-1]
Y = feature.values[:, -1]

# X_scale = np.c_[X[:, 0], preprocessing.scale(X)[:, 1:]]
X_scale = preprocessing.scale(X)[:, 0:]
pca = PCA(n_components=26)
pca.fit(X_scale)
X_scale_pca = pca.transform(X_scale)

# # -------------------- training -----------------#
model = MLPClassifier(hidden_layer_sizes=(16), alpha=0.02, activation='logistic',
                      solver='lbfgs', random_state=0, max_iter=350, early_stopping=True,
                      verbose=False, epsilon=1e-04
                      )

model.fit(X_scale_pca, Y)

# exit(-1)
# ------------------- validation ------------- #
# Y_predict = model.predict(XX_selected)
# Y_predict = model.predict(XX)

# print(Y_predict)
# cnt = 0
#
# for i,j in zip(YY, Y_predict):
#     if i==j and j==0:
#         cnt += 1
#
# predict = float(cnt) / (len(Y_predict) - sum(Y_predict))
# recall = float(cnt) / (len(YY) - sum(YY))
# F = 5 * predict * recall / (2 * predict + 3* recall) * 100
#
# print(predict, recall, F)
#

# ------------- prediction ----------------#
dataframe_prediction = pd.read_csv("data/dsjtzs_txfz_testB.txt", delim_whitespace=True, header=None,
                            names=['id', 'sequence', 'target'])  # load dataset
# XX, dids = feature_selection(dataframe_prediction, 'prediction')

# ------------- pre prediction ----------------#
XX = pd.read_csv('./data/prediction_feature.csv', delim_whitespace=True, header=None).values
dids = pd.read_csv('./data/prediction_filter_id.csv', delim_whitespace=True, header=None).values

scaler = preprocessing.StandardScaler().fit(X)
# XX_scale = np.c_[XX[:, 0], scaler.transform(XX)[:, 1:]]
XX_scale = scaler.transform(XX)[:, 0:]
XX_scale_pca = pca.transform(XX_scale)


ids = np.delete(dataframe_prediction['id'].values, dids)
yy_predict = model.predict(XX_scale_pca)

np.savetxt('dsjtzs_txfzjh_preliminary.txt', ids[yy_predict == 0].astype(int), fmt='%s')
