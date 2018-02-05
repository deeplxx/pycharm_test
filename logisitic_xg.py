import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 对率函数
#
def sigmoid(i):
    return 1 / (1 + np.exp(-i))


# 梯度上升GD
#
def grad_ascent(datamat, labelmat):
    m, n = np.shape(datamat)
    alpha = 0.1  # 步长
    max_cycles = 500  # 迭代次数
    weight = np.mat(np.ones(n))  # 回归系数

    for k in range(max_cycles):
        a = np.dot(datamat, weight.transpose())  # 更新点x
        h = sigmoid(a)  # 更新后的f(x)
        error = (labelmat - h)  # (y - f(x))
        # noinspection PyTypeChecker
        weight += alpha * np.dot(datamat.transpose(), error).transpose()

    return weight


# 随机梯度上升SGD
#
def stochastic_grad_ascent(datamat, labelmat, num_iter=200):
    m, n = np.shape(datamat)
    # alpha = 0.01
    weight = np.mat(np.ones(n))
    for j in range(num_iter):
        data_index = np.arange(m).tolist()
        for i in range(m):
            alpha = 4.0 / (1.0 + j + i) + 0.01  # 避免步长严格下降也常见于模拟退火中！！！
            # noinspection PyTypeChecker
            rand_index_index = int(np.random.uniform(0, len(data_index)))
            rand_index = data_index[rand_index_index]

            h = sigmoid(sum(np.dot(datamat[rand_index], weight.transpose())))  # 想想当时的公式！！！
            error = labelmat[rand_index] - h
            weight += alpha * error * datamat[rand_index]

            del(data_index[rand_index_index])

    return weight


# LDA
#
def lda(data):
    labelmat_1 = data[data.label == 1]
    labelmat_0 = data[data.label == 0]
    x1 = labelmat_1.values[:, :2]  # 正例样本
    x0 = labelmat_0.values[:, :2]  # 反例样本
    mean1 = np.array((np.mean(x1[:, 0]), np.mean(x1[:, 1])))  # 正例均值向量
    mean0 = np.array((np.mean(x0[:, 0]), np.mean(x0[:, 1])))  # 反例均值向量
    s_w = np.array(np.ones((2, 2)))

    for i in range(np.shape(x1)[0]):
        xmean = np.mat(x1[i, :] - mean1)
        s_w += xmean.transpose() * xmean

    for i in range(np.shape(x0)[0]):
        xmean = np.mat(x0[i, :] - mean0)
        s_w += xmean.transpose() * xmean

    weight = (mean0 - mean1) * np.mat(s_w).I
    return weight


# 画图
#
def plotbestfit(datamat, labelmat, weight):
    m = np.shape(datamat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(m):
        if labelmat[i] == 1:
            xcord1.append(datamat[i, 1])
            ycord1.append(datamat[i, 2])
        else:
            xcord2.append(datamat[i, 1])
            ycord2.append(datamat[i, 2])

    plt.figure(1)
    ax1 = plt.subplot(111)
    ax1.scatter(xcord1, ycord1, s=30, c='r', marker='s')
    ax1.scatter(xcord2, ycord2, s=30, c='g')

    x = np.arange(0.2, 0.8, 0.1)
    y = np.array((-weight[0, 0] - weight[0, 1] * x) / weight[0, 2])  # w1 * x + w2 * y + w0 = 0, 0是sigmoid函数的输入！！！
    # y = np.array((-weight[0, 0] * x) / weight[0, 1])
    plt.sca(ax1)
    plt.plot(x, y)
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.title('lda')
    plt.show()


# xg_Path = unicode(xg_Path, 'GB2312')
xg_Data = pd.read_excel('C:/Users/sumlo/Documents/Python_Study/xg.xlsx', header=None)

xg_Data.index = ['density', 'ratio_sugar', 'g', 'label']
xg_Data = xg_Data.transpose()
xg_Data['norm'] = 1
dataMat = np.mat(xg_Data[['norm', 'density', 'ratio_sugar']].values)
labelMat = np.mat(xg_Data['label'].values).transpose()
# noinspection PyTypeChecker

weights = grad_ascent(dataMat, labelMat)
# weights = lda(xg_Data)
# weights = stochastic_grad_ascent(dataMat, labelMat, 200)
plotbestfit(dataMat, labelMat, weights)
