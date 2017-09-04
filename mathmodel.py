#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/30 15:29
# @Author  : Kiristingna
# @Site    : 
# @File    : mathmodel.py
# @Software: PyCharm

# import networkx as nx
# from networkx.algorithms.flow import preflow_push
# import matplotlib.pyplot as plt
#
# G = nx.DiGraph()
# G.add_edge('s','a', capacity=28.0)
# G.add_edge('s','b', capacity=7.0)
# G.add_edge('c','a', capacity=7.0)
# G.add_edge('b','c', capacity=12.0)
# G.add_edge('a','b', capacity=6.0)
# G.add_edge('s','c', capacity=19.0)
#
# G.add_edge('a','d', capacity=15.0)
# G.add_edge('c','d', capacity=14.0)
# G.add_edge('e','b', capacity=10.0)
# G.add_edge('c','t', capacity=36.0)
# G.add_edge('d','e', capacity=7.0)
# G.add_edge('e','t', capacity=18.0)
# G.add_edge('d','t', capacity=23.0)
# colors = [1,1,3,3,3,2,2]
# pos=nx.spring_layout(G)
# # 返回残留网络
# R = preflow_push(G, 's', 't')
# flow_value = nx.maximum_flow_value(G, 's', 't')
# nx.draw_networkx_nodes(G, pos , node_color=colors)
# nx.draw_networkx_edges(G, pos)
# nx.draw_networkx_labels(G, pos)
# nx.draw_networkx_edge_labels(G, pos)
#
# plt.show()
#
# print(flow_value)
# print(R.graph['flow_value'])

# 线性规划
# linprog(c, a, b, eq_x, eq_y, bounds)
# problem min:  c ^ T * X
#         s.t.  a * X < b
#               eq_x * X = eq_y
#               bounds_low <= x1, x2, x3...<=bounds_up
# from scipy.optimize import linprog
# import numpy as np
#
# c = np.array([2, 3, -5])
# a = np.array([[-2, 5, -1], [1, 3, 1]])
# b = np.array([-10, 12])
#
# result = linprog(-c, a, b, [[1,1,1]], [7], bounds=((0, 7), (0, 7), (0, 7)))
# print(result)

import pandas as pd
import numpy as np
dataframe_1 = pd.read_csv('dsjtzs_txfzjh_preliminary.txt', delim_whitespace=True, header=None, names=['id'])
dataframe_2 = pd.read_csv('ss.txt', delim_whitespace=True, header=None, names=['id'])

datas = np.union1d(dataframe_1['id'], dataframe_2['id'])

np.savetxt('combine.txt', datas, fmt='%s')