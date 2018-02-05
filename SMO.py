# import
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Format_Data import format_trans_smo


# 随机选取第二个变量(高效的选取步长最大的j)
#
def selectj(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))

    return j


# 修剪原始解
#
def clip_alpha(alpha, h, l):
    if alpha > h:
        alpha = h
    elif alpha < l:
        alpha = l

    return alpha


# 简单smo
#
def smo_simple(dataset, labelset, c, toler, max_iter):
    datamat = np.mat(dataset)
    labelmat = np.mat(labelset).transpose()
    b = 0  # 初始化b
    m, n = np.shape(datamat)
    alphas = np.zeros((m, 1))  # 初始化alpha
    iter_count = 0  # 没有任何alpha改变的情况下的遍历次数

    while iter_count < max_iter:
        alphapairs_changed = 0  # 优化了的alpha对的数量

        for i in range(m):
            f_i = float(np.multiply(alphas, labelmat).T * (datamat * datamat[i, :].T)) + b
            error_i = f_i - labelmat[i]

            if ((labelmat[i] * error_i < -toler) and (alphas[i] < c)) or \
                    ((labelmat[i] * error_i > toler) and (alphas[i] > 0)):
                j = selectj(i, m)
                k11 = datamat[i, :] * datamat[i, :].T
                k22 = datamat[j, :] * datamat[j, :].T
                k12 = datamat[i, :] * datamat[j, :].T
                f_j = float(np.multiply(alphas, labelmat).T * (datamat * datamat[j, :].T)) + b
                error_j = f_j - labelmat[j]

                # 保存old
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                # 计算H和L
                if labelmat[i] != labelmat[j]:
                    low = max(0, alpha_j_old - alpha_i_old)
                    high = min(c, c + alpha_j_old - alpha_i_old)
                else:
                    low = max(0, alpha_i_old + alpha_j_old - c)
                    high = min(c, alpha_j_old + alpha_i_old)
                if low == high:
                    print('low = high')
                    continue

                # 更新ai和aj
                eta = k11 + k22 - 2.0 * k12
                if eta <= 0:  # 返回有待商榷
                    continue
                alphas[j][0] += labelmat[j] * (error_i - error_j) / eta
                alphas[j][0] = clip_alpha(alphas[j], high, low)
                if abs(alphas[j] - alpha_j_old) <= 0.001:
                    print('j not moving enough')
                    continue
                alphas[i][0] += labelmat[i] * labelmat[j] * (alpha_j_old - alphas[j])

                # 更新b
                b1 = b - error_i - labelmat[i] * k11 * (alphas[i] - alpha_i_old) - \
                     labelmat[j] * k12 * (alphas[j] - alpha_j_old)
                b2 = b - error_j - labelmat[i] * k12 * (alphas[i] - alpha_i_old) - \
                     labelmat[j] * k22 * (alphas[j] - alpha_j_old)
                if 0 < alphas[i] < c:
                    b = b1
                elif 0 < alphas[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphapairs_changed += 1
                print('iter:%d   i:%d, pairs changed %d' % (iter_count, i, alphapairs_changed))

        if alphapairs_changed == 0:
            iter_count += 1
        else:
            iter_count = 0
        print('iter_count:%d' % iter_count)

    return alphas, b


# 画点
#
def plot_data(datamat, labelmat, alphas):
    datamat = np.mat(datamat)
    m = np.shape(datamat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    xcord3 = []
    ycord3 = []

    for i in range(m):
        if labelmat[i] == 1:
            xcord1.append(datamat[i, 0])
            ycord1.append(datamat[i, 1])
        else:
            xcord2.append(datamat[i, 0])
            ycord2.append(datamat[i, 1])

    for i in range(len(alphas)):
        if alphas[i] > 0:
            xcord3.append(datamat[i, 0])
            ycord3.append(datamat[i, 1])

    plt.figure(1)
    ax1 = plt.subplot(111)
    ax1.scatter(xcord1, ycord1, s=30, c='r', marker='s')
    ax1.scatter(xcord2, ycord2, s=30, c='g')
    ax1.scatter(xcord3, ycord3, s=30, c='b')
    plt.show()


xg_Data = pd.read_excel('C:/Users/sumlo/Documents/Python_Study/xg.xlsx', header=None)
data_set = xg_Data.ix[:1].values
label_set = xg_Data.ix[2].values

data_list, label_list = format_trans_smo(data_set, label_set)
a, b_ = smo_simple(data_list, label_list, 0.6, 0.001, 40)
plot_data(data_list, label_set, a)
