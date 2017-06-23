#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/5/24 16:23
# @Author  : Kiristingna
# @Site    : 
# @File    : model.py
# @Software: PyCharm
# from keras.models import model_from_json

import numpy as np
import pandas as pd
from util.split import *
from util.calc import *

# 计算熵
def Shannon(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent


def feature_selection(dataframe, mode):
    Y = None
    if mode == 'train':
        # category data
        Y = dataframe.cate.values

    # feature data
    feature = []
    row_index_delete = []

    # split data to array
    for u, data in dataframe.iterrows():
        # id
        _id = int(data['id'])

        # split str
        p = splits(data['sequence'])

        # filter: 针对小于10个点的数据 直接删除
        if len(p) <= 3:
            row_index_delete.append(u)
            continue

        # 目标点横坐标
        # _target_x = float(data['target'].split(',')[0])

        # 计算差分
        _diff = diff_points(p)
        # 时间间隔
        time_overlap = cal_time(_diff)

        tt, speed_x, speed_y = cal_speed(_diff, p)  # tt, speed = cal_speed_ma(_diff, p, 5) # 滑动平均


        x_acc = np.divide(np.diff(speed_x), tt[1:]) # x轴加速度
        # y_acc = np.divide(np.diff(speed_y), tt[1:]) # y轴加速度

        # 斜边
        s = calcu_s(_diff[:, 0], _diff[:, 1])
        #  w ??
        # w = calcu_w_func(_diff[:, 0], _diff[:, 1])
        thita = calcu_thita(_diff[:, 0], _diff[:, 1], s)

        # ---------- ent feature --------- #
        x_ent = calcu_entropy_func(_diff[:, 0])  # dx 分布熵
        y_ent = calcu_entropy_func(_diff[:, 1])  # dy 分布熵


        # ---------- time feature --------- #
        time_max = np.log(np.max(np.fabs(time_overlap)))  # 时间间隔最大值的log
        time_total = np.log(calcu_passtime(np.fabs(time_overlap)))  # 经过的总时间的log
        strange_t = calcu_strange_t(time_overlap)  # 时间间隔的离群点 +人工参数 ratio+
        strange_vx = calcu_strange_vx(speed_x)
        time_tail_2 = calcu_tail(time_overlap)
        t_skewness = calcu_skewness(time_overlap)
        t_kurtosis = calcu_kurtosis(time_overlap)

        # ----------- continous feature ------#
        y_max_continous = calcu_y_max_continous(_diff[:, 1])

        # ----------- speed feature ---------- #
        speed_x_var = calcu_xv_var(_diff[:, 0], _diff[:, 2])

        # ----------- angle ----------------- #
        th_std = calcu_thita_std(thita)

        # 不规则feature
        passtime_steps_ratio = calcu_passtime_steps_ratio(time_overlap)

        # w_kurtosis = calcu_kurtosis(w)
        # reverse_ax = calcu_reverse_ax(speed_x, tt)
        # ------------ 待加入 -----------------#

        #---- w ----#
        # x_var = np.var(_diff[:, 0])

        # 加速度最大值
        x_acc_max = np.max(np.fabs(x_acc))

        # ------------- 待加入结束 --------------#

        # 特征集合
        feature.append(np.array([
                                 x_ent,
                                 y_ent,
                                 # w[0], w[-1],
                                 speed_x_var,
                                 y_max_continous,
                                 time_max, time_tail_2,
                                 strange_t,strange_vx,
                                 passtime_steps_ratio,
                                 t_skewness, t_kurtosis,
                                 # reverse_ax,
                                 th_std,
                                 x_acc_max
                                 ]))

    if mode == 'train':
        Y = np.delete(Y, row_index_delete)
        return np.array(feature), Y, row_index_delete
    elif mode == 'prediction':
        return np.array(feature), row_index_delete

if __name__ == "__main__":
    dataframe = pd.read_csv("../data/dsjtzs_txfz_training.txt", delim_whitespace=True, header=None,
                            names=['id', 'sequence', 'target', 'cate'])  # load dataset
    X, Y, dids = feature_selection(dataframe, 'train')
    np.savetxt('../result1.csv', np.c_[X, Y], fmt="%s")