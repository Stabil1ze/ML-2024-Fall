import numpy as np
import os
import math
import matplotlib.pyplot as plot


def phi_x(x):
    phi_row = [1]
    for j in range(1, 25):
        phi_row.append(math.e ** (-(x - 0.2 * (j - 12.5)) ** 2))
    return np.array(phi_row)


def hat_w(file_name, re_co):
    f = open(file_name, "r")
    # 计算Phi和y_l
    phi = np.zeros((25, 25))
    x_l = np.zeros((25, 1))
    y_l = np.zeros((25, 1))
    i = 0
    for point_pair in f.readlines():
        point_pair = point_pair.strip().split()
        x = float(point_pair[0])
        y = float(point_pair[1])
        phi[i] = phi_x(x).T
        x_l[i] = x
        y_l[i] = y
        i += 1
    f.close()
    w = np.linalg.inv(np.dot(phi.T, phi) + re_co * np.eye(25))
    w = np.dot(np.dot(w, phi.T), y_l)
    return w


def y_bar(x, w):
    return float(np.inner(w.T, phi_x(x)))


# 计算bias
def bias(file_list, w):
    bias_sum = 0
    for file_name in file_list:
        f = open(file_name, "r")
        for point_pair in f.readlines():
            point_pair = point_pair.strip().split()
            x = float(point_pair[0])
            bias_sum += (y_bar(x, w) - math.sin(math.pi * x)) ** 2
        f.close()
    bias_sum = bias_sum / 2500
    return bias_sum


# 计算variance
def variance(file_list, w, re_co):
    print(re_co)
    variance_sum = 0
    hat_w_list = {}
    # 计算每个文件的hat_w以便调用
    for file_name in file_list:
        hat_w_list[file_name] = hat_w(file_name, re_co)
    for file_name in file_list:
        f = open(file_name, "r")
        # 遍历每个点
        for point_pair in f.readlines():
            point_pair = point_pair.strip().split()
            x = float(point_pair[0])
            # 计算每个点的variance
            y_bar_x = y_bar(x, w)
            this_phi = phi_x(x)
            for file_name_2 in file_list:
                y_l_x = np.inner(hat_w_list[file_name_2].T, this_phi)
                variance_sum += (y_l_x - y_bar_x) ** 2
        f.close()
    variance_sum = variance_sum / 250000
    return variance_sum


# 遍历所有文件，计算平均估计函数y_bar的系数矩阵w_bar
def compute_w_bar(file_list, re_co):
    w_bar = np.zeros((25, 1))
    for file_name in file_list:
        w_bar += hat_w(file_name, re_co)
    w_bar = w_bar / 100
    return w_bar


co_range = np.arange(-3, 1.1, 0.1)
co_list = [10 ** float(i) for i in co_range]
data_list = []
w_bar_dict = {}
for root, dirs, files in os.walk("Exercise-5-data"):
    for file in files:
        data_list.append("Exercise-5-data\\" + file)
plot.figure()
for co in co_list:
    w_bar_dict[co] = compute_w_bar(data_list, co)
list_bias = [bias(data_list, w_bar_dict[co])
             for co in co_list]
list_variance = [variance(data_list, w_bar_dict[co], co)
                 for co in co_list]
plot.plot(co_range, list_bias, label="bias^2")
plot.plot(co_range, list_variance, label="variance")
plot.plot(co_range, [list_bias[i] + list_variance[i] for i in range(len(list_bias))], label="bias^2 + variance")
plot.legend()
plot.show()
