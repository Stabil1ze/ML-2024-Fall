import numpy as np
import matplotlib.pyplot as plt
import math


def phi_x(x):
    phi_row = [1]
    for j in range(1, 25):
        phi_row.append(math.e ** (-(x - 0.2 * (j - 12.5)) ** 2))
    return np.array(phi_row)


def hat_w(phi, y_l, re_co):
    w = np.linalg.inv(np.dot(phi.T, phi) + re_co * np.eye(25))
    w = np.dot(np.dot(w, phi.T), y_l)
    return w


# 对文件data_i，计算给定系数下的\hat{w}
def compute_parameter_estimator(file_name, re_co_list):
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

    # 计算\hat{w}并画图
    plt.figure(figsize=(10, 8))
    plt.scatter(x_l, y_l, marker='o', color='black')
    x_range = np.arange(-1, 1, 0.01)
    for re_co in re_co_list:
        plt.plot(x_range, [np.dot(hat_w(phi, y_l, re_co).T, phi_x(x)) for x in x_range],
                 label=r"$\lambda$ = " + str(re_co))
    plt.legend()
    plt.show()


co_list = [10, 0.1, 1e-5, 1e-10]
file = "Exercise-5-data\\data_1"
compute_parameter_estimator(file, co_list)

