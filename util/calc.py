#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/18 10:19
# @Author  : Kiristingna
# @Site    : 
# @File    : calc.py
# @Software: PyCharm
import numpy as np

def calc_five_point(point, length, idxx):
    xn = length
    for i in range(5):
        if idxx[i] < 0:
            idxx[i] = 0
        if idxx[i] > (xn - 1):
            idxx[i] = xn - 1

    dt = point[idxx[-1]][2] - point[idxx[0]][2]
    dx = point[idxx[-1]][0] - point[idxx[0]][0]
    dy = point[idxx[-1]][1] - point[idxx[0]][1]
    mt = point[-1][2]

    dt = dt if dt > 1e-5 else 4200.0
    mt = mt if mt > 1e-5 else 700.0
    a = dx / dt
    b = dy / dt
    c = dt / mt
    return [a, b, c]

def cal_distance(point):
    x = point[:, 0]
    y = point[:, 1]
    dist_diff = np.sqrt(x ** 2 + y ** 2)
    return dist_diff

def cal_angle(points):
    pass

def cal_toward(point, length):
    n = length
    x = point[:, 0]
    y = point[:, 1]
    t = point[:, 2]

    twz = np.zeros([3, 3], dtype='float')
    for i in range(1, n - 1):
        dt = t[i] - t[i - 1]
        if x[i] - x[i - 1] == 0:
            if y[i] - y[i - 1] == 0:
                twz[0, 0] += dt
            elif y[i] - y[i - 1] > 0:
                twz[0, 1] += dt
            else:
                twz[0, 2] += dt
        elif x[i] - x[i - 1] > 0:
            if y[i] - y[i - 1] == 0:
                twz[1, 0] += dt
            elif y[i] - y[i - 1] > 0:
                twz[1, 1] += dt
            else:
                twz[1, 2] += dt
        else:
            if y[i] - y[i - 1] == 0:
                twz[2, 0] += dt
            elif y[i] - y[i - 1] > 0:
                twz[2, 1] += dt
            else:
                twz[2, 2] += dt

    # pass
    twz = twz / float(t[-1])
    # twz=twz**0.5
    return twz.reshape([1, 9])[0]


def cal_plr(point, length):
    n = length
    x = point[:, 0]
    y = point[:, 1]
    t = point[:, 2]

    twz = np.zeros([2, 3], dtype='float')
    for i in range(1, n - 1):
        dt = t[i] - t[i - 1]
        if x[i] - x[i - 1] == 0:
            twz[0, 0] += dt
        elif x[i] - x[i - 1] > 0:
            twz[0, 1] += dt
        else:
            twz[0, 2] += dt

        if y[i] - y[i - 1] == 0:
            twz[1, 0] += dt
        elif y[i] - y[i - 1] > 0:
            twz[1, 1] += dt
        else:
            twz[1, 2] += dt

    twz = twz / t[-1]
    # twz=twz**0.5
    return twz.reshape([1, 6])[0]