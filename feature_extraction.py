#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/5/24 16:23
# @Author  : Kiristingna
# @Site    : 
# @File    : feature_extraction.py
# @Software: PyCharm

from util.split import *
from util.calc import *


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

        # 计算差分
        _diff = diff_points(p)
        # 时间间隔
        time_overlap = cal_time(_diff)

        speed_x = cal_speed(_diff)
        # speed_x_log = np.log1p(speed_x)
        speed_x_diff = np.diff(speed_x)
        x_y = cal_distance(_diff)

        # ---------- position feature --------- #
        start_x_pos = p[0][0]  # 起始的x位置 :feature:
        start_y_pos = p[0][1]  # 起始的y位置 :feature:
        x_y_stop = len(x_y[x_y == 0])  # 停留点个数 :feature:

        len_track = 3  # 轨迹点的个数

        start_t = p[0][2]  # 起始的t :feature:
        t_min = time_overlap.min()  # :feature: 最小间隔
        t_max = time_overlap.max()  # :feature:
        t_var = time_overlap.var()  # tDe std :feature:
        t_mean = time_overlap.mean()
        t_stop = len(time_overlap[time_overlap==0])  # :feature:
        end_t = p[-1][2]  # 结束的t :feature:
        # ---------- end position feature --------- #

        # 目标点坐标
        end_x_pos = p[-1][0]
        end_y_pos = p[-1][1]
        _target_x = float(data['target'].split(',')[0])
        _target_y = float(data['target'].split(',')[1])

        # ---------- distance feature --------- #
        dur_x = end_x_pos - _target_x # :feature:
        dur_y = end_y_pos - _target_y # :feature:
        y_last_init = end_y_pos - start_y_pos
        distance = np.sqrt(dur_x ** 2 + dur_y**2) # :feature:
        # ---------- end distance feature --------- #

        det = p[:, 0] -_target_x
        dst_min = det.min()
        dst_max = det.max()
        dst_mean = det.mean()

        #----------- speed and x y------------------#
        dx_min = np.min(_diff[:, 0])
        dx_max = np.max(_diff[:, 0])
        dx_mean = np.mean(_diff[:, 0])
        sx_min = np.min(p[:, 0])
        sx_max = np.max(p[:, 0])
        y_min = np.min(p[:, 1])
        speed_init = speed_x[0]
        x_min = np.nanmin(speed_x)
        x_max = np.nanmax(speed_x)
        x_mean = np.nanmean(speed_x)
        if len(speed_x_diff) > 0:
            speed_min = np.nanmin(speed_x_diff)
            speed_max = np.nanmax(speed_x_diff)
            speed_var = np.var(speed_x_diff)
            speed_median = np.nanmedian(speed_x_diff)
            speed_mean = np.nanmean(speed_x_diff)
        else:
            speed_min = 0
            speed_max = 0
            speed_var = 0
            speed_median = 0
            speed_mean = 0
        #------------end speed -------------#

        # ---------- back numbers --------------#
        is_go_back = 0
        for i in range(len(p) - 1):
            if (p[i + 1][0] < p[i][0]):
                is_go_back = 1
        # ---------- end back numbers ----------#

        # ------- densitity --------------- #
        if (np.max(p[:, 0]) - np.min(p[:, 0])) >= 1:
            x_density = float(len(p)) / (np.max(p[:, 0]) - np.min(p[:, 0]))
        else:
            x_density = 0
        if (np.max(p[:, 1]) - np.min(p[:, 1])) >= 1:
            y_density = float(len(p)) / (np.max(p[:, 1]) - np.min(p[:, 1]))
        else:
            y_density = 0

        if (np.max(p[:, 2]) - np.min(p[:, 2])) >= 1:
            t_density = float(len(p)) / (np.max(p[:, 2]) - np.min(p[:, 2]))
        else:
            t_density = 0

        # 中点 取点采样
        mid_pos = int(len_track / 2)
        idxx = list(range(mid_pos-2, mid_pos + 3))
        xyt_1 = calc_five_point(p, len_track, idxx) # 5 点统计 返回 abc

        idxx = list(range(0, 5))
        xyt_2 = calc_five_point(p, len_track, idxx) # 5 点统计 返回 abc

        idxx = list(range(len_track - 6, len_track - 1))
        xyt_3 = calc_five_point(p, len_track, idxx) # 5 点统计 返回 abc

        #?
        twz = cal_toward(p, len_track)
        plr = cal_plr(p, len_track)

        # 特征集合
        _f = np.hstack((np.array([
            start_x_pos,
            start_y_pos,
            x_y_stop,
            is_go_back,
            end_t, start_t, t_stop,
            t_min, t_max, t_var, t_mean,
            dur_x, dur_y, distance,
            y_last_init,
            dst_min, dst_max, dst_mean,
            y_min, sx_min, sx_min/sx_max, x_density, y_density, t_density,
            speed_init, speed_min, speed_max, speed_var, speed_median, speed_mean,
            x_min, x_max, x_mean,
            dx_min, dx_max, dx_mean,
        ]),
                        np.array(xyt_1),
                        np.array(xyt_2),
                        np.array(xyt_3),
                        np.array(twz),
                        np.array(plr)
        ))

        feature.append(_f)
    if mode == 'train':
        Y = np.delete(Y, row_index_delete)
        return np.array(feature), Y, row_index_delete
    elif mode == 'prediction':
        return np.array(feature), row_index_delete

if __name__ == "__main__":
    dataframe = pd.read_csv("./data/dsjtzs_txfz_training.txt", delim_whitespace=True, header=None,
                            names=['id', 'sequence', 'target', 'cate'])  # load dataset
    X, Y, dids = feature_selection(dataframe, 'train')
    np.savetxt('./data/train_feature.csv', np.c_[X, Y], fmt="%s")

    dataframe_prediction = pd.read_csv("data/dsjtzs_txfz_testB.txt", delim_whitespace=True, header=None,
                                       names=['id', 'sequence', 'target'])  # load dataset
    XX, dids = feature_selection(dataframe_prediction, 'prediction')
    np.savetxt('./data/prediction_feature.csv', XX, fmt="%s")

    np.savetxt('./data/prediction_filter_id.csv', dids, fmt="%s")