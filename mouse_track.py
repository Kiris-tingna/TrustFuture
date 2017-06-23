#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/5/21 18:05
# @Author  : Kiristingna
# @File    : mouse_track.py
# @Software: PyCharm

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pandas as pd
import numpy as np
from util.model import feature_selection

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn import svm
from sklearn.neural_network import MLPClassifier

np.seterr(all='ignore')
# ----------------- sampling --------------#
dataframe_train = pd.read_csv("data/dsjtzs_txfz_training.txt", delim_whitespace=True, header=None,
                            names=['id', 'sequence', 'target', 'cate'])  # load dataset
X, Y, del_ids = feature_selection(dataframe_train, 'train')

X_train, X_test, Y_train, Y_test = train_test_split(X[Y == 1], Y[Y == 1], test_size= 0.001, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(X[Y == 0], Y[Y == 0], test_size= 0.001, random_state=1)

# X_sampled = np.concatenate((X_train, x_train))
# Y_sampled = np.concatenate((Y_train, y_train))
# 调整样本
# X_sampled = np.concatenate((X_train, x_train, x_train))
# Y_sampled = np.concatenate((Y_train, y_train, y_train))

# XX = np.concatenate((X_test, x_test))
# YY = np.concatenate((Y_test, y_test))
X_sampled = X_train
Y_sampled = Y_train

# -------------------- training -----------------#
# model = AdaBoostClassifier(n_estimators=100, learning_rate=1.8, algorithm='SAMME.R')
# model = svm.SVC(kernel='rbf') # C值小 泛华能力强
model = svm.OneClassSVM(nu=0.18, kernel="rbf", gamma=0.05)
# model = RandomForestClassifier(n_estimators=80, n_jobs=-1, oob_score=True)
# model = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.2, subsample=0.5)
# model = MLPClassifier(hidden_layer_sizes=(80, 20), alpha=0.5,
#                       verbose=True, tol=1e-4, random_state=1,
#                       learning_rate_init=0.01)


# svm特征预处理
X_selected = MinMaxScaler().fit_transform(X_sampled)
# X_selected = Normalizer().fit_transform(X_sampled)

# XX_selected = MinMaxScaler().fit_transform(XX)
# XX_selected = Normalizer().fit_transform(XX_selected)
model.fit(X_selected, Y_sampled)


# model.fit(X_sampled, Y_sampled)
# print(model.feature_importances_)
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
dataframe_prediction = pd.read_csv("data/dsjtzs_txfz_test1.txt", delim_whitespace=True, header=None,
                            names=['id', 'sequence', 'target'])  # load dataset
XX, dids = feature_selection(dataframe_prediction, 'prediction')
XX_selected = MinMaxScaler().fit_transform(XX)
# XX_selected = Normalizer().fit_transform(XX)

ids = np.delete(dataframe_prediction['id'].values, dids)
yy_predict = model.predict(XX_selected)
np.savetxt('dsjtzs_txfzjh_preliminary.txt', ids[yy_predict == -1].astype(int), fmt='%s')
