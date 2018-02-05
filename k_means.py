import numpy as np
# import pandas as pd


# 样本间距离
#
def dist_arr(vec1, vec2):
    return np.sqrt(sum(np.power(vec1 - vec2, 2)))


# 初始化原型向量（只支持连续属性）
#
def rand_cent_arr(datamat, k):
    n = np.shape(datamat)[1]
    centroids = np.ones((k, n))

    for i in range(n):
        min_i = min(datamat[:, i])
        max_i = max(datamat[:, i])
        rang_i = max_i - min_i
        centroids[:, i] = min_i + rang_i * np.random.rand(k)

    return centroids


# k均值
#
def k_means(datamat, k, dist_func=dist_arr, cent_func=rand_cent_arr):
    m, n = np.shape(datamat)
    cluster_changed = True  # 原型向量更新与否的标记
    cluster_assment = np.zeros((m, 2))  # 每个样本对应的簇标记以及离中心店的距离平方
    centriods = cent_func(datamat, k)  # 初始化原型向量（簇质心）

    # 若原型向量有更新则重复
    while cluster_changed:

        for i in range(m):
            cluster_changed = False
            min_dist = np.inf
            class_index = -1  # 簇标记

            # 更新簇标记以及距离
            for j in range(k):
                dist_i = dist_func(datamat[i], centriods[j])
                if dist_i < min_dist:
                    min_dist = dist_i
                    class_index = j

            # 若任意簇标记有变化则原型向量有变化
            if cluster_assment[i, 0] != class_index:
                cluster_changed = True
            cluster_assment[i, :] = class_index, min_dist ** 2  # 此处用平方是为了方便算SSE（一个值）（平方意味着更重视远处的点）

        # 更新簇质心位置
        for i in range(k):
            centriods[i, :] = np.mean(datamat[cluster_assment[:, 0] == i], axis=0)

    return centriods, cluster_assment


# 测试
# xg_data = pd.read_excel('D:/work/source/xg3.0a.xlsx')
# data_mat = xg_data.values[:, 1:3]
#
# centri, clus = k_means(data_mat, 4)
