#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/12 13:42
# @Author  : Kiristingna
# @Site    : 
# @File    : split.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.arma import *

# 定位某一行
def locate(id: int):
    seq = dataframe['sequence'].iloc[id - 1]
    return splits(seq)

def splits(seq):
    seqs = seq.rstrip(';').split(';')
    points = np.array([_x.split(',') for _x in seqs]).astype(float)
    return points

# 差分计算
def diff_points(points):
   return np.diff(points, axis=0)


# 返回时间点和该时间点的速度
def cal_speed(diff, points):

    speed_x = np.divide(diff[:, 0], diff[:, 2])
    speed_y = np.divide(diff[:, 1], diff[:, 2])
    return filter(diff[:, 2], speed_x, speed_y)

# def cal_speed_ma(diff, points, step):
#     tt, speed = cal_speed(diff, points)
#     return moving_average(step, tt, speed)

# 计算时间
def cal_time(diff):
    return diff[:, 2]

# 删去nan值
def filter(time, arr, arr2):
    _l = len(arr)
    index = []

    for idx in range(_l):
        if np.isinf(arr[idx]) or np.isinf(arr2[idx]):
            index.append(idx) # delete
        elif np.isnan(arr[idx]) or np.isnan(arr2[idx]):
            arr[idx] = 0
            arr2[idx] = 0

    time = np.delete(time, index)
    arr = np.delete(arr, index)
    arr2 = np.delete(arr2, index)

    return time, arr, arr2

def plot(id, tt, yy):
    plt.clf()
    plt.plot(tt, yy)
    # plt.savefig("../machine/{}_ma.png".format(id))
    plt.show()

if __name__ == "__main__":
    dataframe = pd.read_csv("../data/dsjtzs_txfz_training.txt", delim_whitespace=True, header=None,
                            names=['id', 'sequence', 'target', 'cate'])  # load dataset
    for id in range(1, 200):
        _diff = diff_points(locate(id))
        _time = cal_time(_diff)
        xx = range(0, len(_time))
        plt.clf()
        plt.scatter(xx, _time)
        plt.savefig("../human/{}_time_overlap.png".format(id))


