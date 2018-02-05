from Format_Data import *
import pandas as pd
import numpy as np

# 单层决策树生成函数(学习器) #######


# 将连续变量二值化处理
#
def stump_classify(datamat, dimen, thresh_val, thresh_ineq):
    ret_array = np.ones((np.shape(datamat)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[datamat[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[datamat[:, dimen] > thresh_val] = -1.0

    return ret_array


# 构建最优树桩
#
def build_stump(datamat, labelmat, d, num_steps=10):
    """
    
    :param d: 列向量
    :param datamat: 样本
    :param labelmat: 标签
    :param num_steps: 区域数 
    :return: 最优树桩，最小误差向量，最佳预测结果
    """

    m, n = np.shape(datamat)
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf

    # 在所有特征上遍历
    for i in range(n):
        range_min = datamat[:, i].min()
        range_max = datamat[:, i].max()
        step_size = (range_max - range_min) / num_steps

        # 对每个分割区域
        for j in range(-1, num_steps + 1):

            # 对每个方向
            for ineqal in ['lt', 'gt']:
                thresh_val = range_min + j * step_size  # 阈值
                predict_val = stump_classify(datamat, i, thresh_val, ineqal)  # 预测标记
                errormat = np.mat(np.ones((m, 1)))  # 错误矩阵
                errormat[predict_val == labelmat] = 0
                weight_error = d.T * errormat  # 错误率

                # print("split: dim %d, thresh %.2f, thresh ineqal %s, weight error %.3f" % (i, thresh_val, ineqal,
                #                                                                            weight_error))

                if weight_error < min_error:
                    min_error = weight_error
                    best_class_est = predict_val.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineqal'] = ineqal

    return best_stump, min_error, best_class_est


# 基于单层决策树的adaboost训练过程
#
def adaboost_train_ds(datamat, labelmat, num_iter=40):
    weak_class_list = []  # 弱学习器列表
    m = np.shape(datamat)[0]
    d = np.mat(np.ones((m, 1)) / m)
    agg_class_est = np.mat(np.ones((m, 1)))  # 每个样本的类别估计累计值， H

    for i in range(num_iter):
        best_stump, error, class_est = build_stump(datamat, labelmat, d)
        weak_class_list.append(best_stump)
        print("D: ", d.T)
        print("class_est: ", class_est.T)

        # 计算alpha
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # 确保不会出现除0溢出
        best_stump['alpha'] = alpha

        # 更新D
        expon = -1 * alpha * np.multiply(labelmat, class_est)
        d = np.multiply(d, np.exp(expon))
        d = d / d.sum()

        # 评估集成错误率
        agg_class_est += alpha * class_est
        agg_error = np.multiply(np.sign(agg_class_est) != labelmat, np.ones((m, 1)))
        error_rate = agg_error.sum() / m
        print("agg_class_est: ", agg_class_est)
        print("total error: ", error_rate)
        if error_rate == 0.0:
            break

    return weak_class_list


# 分类函数
#
def adaboost_classify(datamat, classifer_list):
    m = np.shape(datamat)[0]
    agg_class_est = np.ones((m, 1))  # H

    # 在所有分类器上遍历
    for i in range(len(classifer_list)):
        class_est = stump_classify(datamat, classifer_list[i]['dim'], classifer_list[i]['thresh'],
                                   classifer_list[i]['ineqal'])
        agg_class_est += classifer_list[i]['alpha'] * class_est

    return np.sign(agg_class_est)


# 导入数据
#
xg_Data = pd.read_excel('C:/Users/sumlo/Documents/Python_Study/xg.xlsx', header=None)
dataset = xg_Data.ix[:1].transpose().values.tolist()
labelset = xg_Data.ix[2].values.tolist()

data_mat, label_mat = format_trans_xg(dataset, labelset)
adaboost_train_ds(data_mat, label_mat)
