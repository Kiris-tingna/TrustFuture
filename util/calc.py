#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/18 10:19
# @Author  : Kiristingna
# @Site    : 
# @File    : calc.py
# @Software: PyCharm
import collections
import numpy as np
import math
FLOAT_EPSILON = 0.00000001

def calcu_entropy_func(nparray):
    counts = collections.Counter(nparray)
    counts = np.array(list(counts.values()),dtype = np.float)
    prob = counts / len(nparray)
    return (-prob * np.log2(prob)).sum()

def calcu_skewness(nparray):
    stdev = np.std(nparray)
    mean = np.mean(nparray)
    mean_element_pow3 = np.mean(nparray ** 3)
    skewness = (mean_element_pow3 - 3*mean*(stdev**2) - mean **3) / (stdev + FLOAT_EPSILON)**3
    return skewness

def calcu_kurtosis(nparray):
    stdev = np.std(nparray)
    mean = np.mean(nparray)
    mean_4 = np.mean((nparray - mean) ** 4)
    kurtosis = mean_4 / (stdev + FLOAT_EPSILON)**4 - 3
    return kurtosis

def calcu_passtime(time):
    return np.sum(time)

# what is this mean?
def calcu_w_func(nparray_x, nparray_y):
    s = np.sqrt(nparray_x**2 + nparray_y**2)
    x1 = nparray_x[:-1]
    x2 = nparray_x[1:]
    y1 = nparray_y[:-1]
    y2 = nparray_y[1:]
    s1 = s[:-1]
    s2 = s[1:]
    w = (x1 * x2 + y1 * y2) / (s1 * s2 + FLOAT_EPSILON)

    return_value = np.arccos(w)
    return return_value


def calcu_passtime_steps_ratio(time):
    result = np.sum(time) / (len(time) * 100)
    return result

def calcu_strange_t(time):
    return strange_points(time, 0.25)

def calcu_strange_vx(speed_x):
    return strange_points(speed_x, 0.4)

def strange_points(array, ratio=1):
    overflow = np.abs(array - np.average(array)) - ratio * np.std(array)
    return np.count_nonzero(np.where(overflow > 0, 1, 0)) / float(len(array))


def calcu_continous_part(array):
    freq = []
    cnt = 1
    for i in range(len(array)-1):
        if array[i + 1] == array[i]:
            cnt += 1
            continue
        if cnt >= 2:
            freq.append(cnt)
    return freq


def calcu_x_continous_scores(dx):
    freq = calcu_continous_part(dx)
    return sum(freq) / float(len(dx))

def calcu_y_continous_scores(dy):
    freq = calcu_continous_part(dy)
    return sum(freq) / float(len(dy))

def calcu_y_max_continous(dy):
    freq = calcu_continous_part(dy)
    if not freq:
        return 0
    return np.max(np.array(freq))

def calcu_xv_var(dx, time):

    time = time[np.where(dx != 0)]
    dx = dx[np.where(dx != 0)]

    if len(dx) ==0:
        return 0
    if np.var(np.fabs(time / dx)) == 0:
        return -10
    else:
        return np.log(np.var(np.fabs(time / dx)))


def calcu_t_peEntropy_2(time):
    return calcu_permutation_entropy(time, 2)


def calcu_x_peEntropy_345(dx):
    x3 = calcu_permutation_entropy(dx, 3)
    x4 = calcu_permutation_entropy(dx, 4)
    x5 = calcu_permutation_entropy(dx, 5)
    return (x3 + x4 + x5) / 3

def calcu_x_2bin_peEntropy234(dx):
    x_bin = calcu_bin(dx, 2)
    x2 = calcu_permutation_entropy(x_bin, 2)
    x3 = calcu_permutation_entropy(x_bin, 3)
    x4 = calcu_permutation_entropy(x_bin, 4)
    return (x2 + x3 + x4) / 3

def calcu_t_2bin_peEntropy234(time):
    t_bin = calcu_bin(time, 2)
    t2 = calcu_permutation_entropy(t_bin, 2)
    t3 = calcu_permutation_entropy(t_bin, 3)
    t4 = calcu_permutation_entropy(t_bin, 4)
    return (t2 + t3 + t4) / 3


# def calcu_vx_2bin_peEntropy(track):
#     vx = track.x / (track.t + 0.01)
#     vx_bin = calcu_bin(vx, 2)
#     vx2 = calcu_permutation_entropy(vx_bin, 2)
#     vx6 = calcu_permutation_entropy(vx_bin, 6)
#     return (vx2 + vx6) / 2


def calcu_cut_values(array, nbin):
    sort_array = sorted(array)
    single_bin_lens = len(array) / nbin
    if single_bin_lens == 0:
        return sort_array
    cut_values = []
    cut_values.append(sort_array[0])
    for i in range(len(array)):
        if (i+1) % single_bin_lens == 0:
            cut_values.append(sort_array[i])
        else:
            continue
    return cut_values


def calcu_bin(array, nbin):
    cut_values = calcu_cut_values(array, nbin)
    array_bin = [value_mapping(val, cut_values, nbin) for val in array]
    return array_bin

def value_mapping(val, cut_values, nbin):
    if val < cut_values[0]:
        return 0.
    for i in range(len(cut_values)-1):
        if cut_values[i] <= val < cut_values[i+1]:
            return i / float(nbin)
    if val >= cut_values[-1]:
        return (len(cut_values)-2) / float(nbin)


def calcu_shape_index(shape):
     shape_len = len(shape)
     index = 0
     for i in range(shape_len):
         index += (shape[i] * (3 ** (shape_len-i-1)))
     return index

def calcu_shape(array):
    n = len(array)
    result = []
    for i in range(n-1):
        result.append(calcu_atom_shape(array[i], array[i+1]))
    return result

def calcu_atom_shape(val1, val2):
    if val1 < val2:
        return 0
    if val1 == val2:
        return 1
    if val1 > val2:
        return 2

def calcu_permutation_entropy(array, n):
    array_len = len(array)
    shape_index = []
    for i in range(array_len-n+1):
        shape_index.append(calcu_shape_index(calcu_shape(array[i:i+n])))
    return calcu_entropy_func(shape_index)

#  加速度倒数
def get_ax_reverse(vx, t):
    ax = np.zeros_like(vx)
    for i in range(len(vx)):
        if i == 0:
            ax[i] = vx[i] / (t[i] + 0.0001)
        else:
            ax[i] = (vx[i] - vx[i - 1]) / (t[i] + 0.0001)
    return ax

def calcu_reverse_ax(speed_x, time):
    vx = np.round(speed_x, 2) * 100
    ax = get_ax_reverse(time, vx)
    var_val = np.var(ax) + 1e-4
    return max(0, np.log(var_val))


def calcu_tail(time):
    t = time[-6:]
    tail_num = max(t) - min(t)
    if tail_num == 0:
        return 0
    return math.log(tail_num, 1.5)

def calcu_s(x, y):
    s = np.sqrt(x ** 2 + y ** 2)
    return s

def calcu_thita(x, y, s):
    s1 = s[:-1]
    s2 = s[1:]

    xx = x[:-1] + x[1:]
    yy = y[:-1] + y[1:]

    s3 = np.sqrt(xx ** 2 + yy ** 2)

    cos_data = (s1 ** 2 + s2 **2 - s3 ** 2) / (2 * s1 * s2)
    np.clip(cos_data, -1, 1)
    thita = np.degrees(np.arccos(cos_data))
    return thita

def calcu_thita_std(thita):
    if np.isnan(np.nanstd(thita)):
        return 0
    else:
        return np.nanstd(thita)