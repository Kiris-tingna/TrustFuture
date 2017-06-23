#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/12 16:59
# @Author  : Kiristingna
# @Site    : 
# @File    : arma.py
# @Software: PyCharm
import numpy as np

# 输入速度数组 滑动窗口大小
def moving_average(step: int, time, speed):
    _speed = []
    _time = []
    _len = len(speed)
    for index in range(0, _len-step):
        sampled_time = time[index : index+step]
        sampled_speed = speed[index : index+step]
        time_step_average = np.nanmean(sampled_time)
        speed_step_average = np.nanmean(sampled_speed)

        _time.append(time_step_average)
        _speed.append(speed_step_average)

    return _time, _speed

