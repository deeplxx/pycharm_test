import pandas as pd
import numpy as np
import pickle_store


# 计算信息熵
#
def cal_ent(data):
    """
    :param data: 数据集
    :return: 该数据集的信息熵
    """
    num_entries = len(data)
    label_counts = {}

    # 给所有可能的分类创建字典
    for feat_vec in data:
        current_label = feat_vec[-1]  # 标签
        if current_label not in label_counts.keys():  # 若不在字典中则添加
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    # 计算ent
    ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        ent -= prob * np.log2(prob)

    return ent


# 划分数据集
#
def split_data(data, axis, value):
    """
    :param data: 数据集
    :param axis: 特征索引
    :param value: axis下的值
    :return: 划分的子样本
    """
    ret_data = []

    for feat_vec in data:
        if feat_vec[axis] == value:
            reduce_feat_vec = feat_vec[:axis]
            reduce_feat_vec.extend(feat_vec[(axis + 1):])
            ret_data.append(reduce_feat_vec)

    return ret_data


# 对连续特征划分数据集
#
def split_continue_data(data, axis, value, direction):
    """
    :param data: 数据
    :param axis: 特征索引
    :param value: axis下的值
    :param direction: 划分方向（1小于value, 0大于value）
    :return: 划分的子样本
    """
    ret_data = []

    for feat_vec in data:
        if (direction == 0 and feat_vec[axis] > value) or (direction == 1 and feat_vec[axis] <= value):
            reduce_feat_vec = feat_vec[:axis]
            reduce_feat_vec.extend(feat_vec[(axis + 1):])
            ret_data.append(reduce_feat_vec)

    return ret_data


# 选择划分特征
#
def choose_bestfeature_to_split(data, feat_list):
    """
    :param data: 数据集
    :param feat_list: 特征列表
    :return:  最优划分特征index,并将连续特征值进行二值化处理
    """
    num_feat = len(data[0]) - 1  # 特征数
    base_ent = cal_ent(data)  # 根节点信息熵
    best_info_gian = 0.0
    best_feat = 0
    best_split_dic = {}  # 连续特征的最佳划分点dic
    info_gain_dic = {}

    for i in range(num_feat):
        feat_i = [example[i] for example in data]  # 第i个特征列
        unique_vals = set(feat_i)

        # 连续特征
        if type(feat_i[0]).__name__ == ('float' or 'int'):

            # 产生n-1个候选划分点
            sort_vals = sorted(feat_i)
            split_list = []  # 候选划分点列表
            for j in range(len(sort_vals) - 1):
                split_list.append((sort_vals[j] + sort_vals[j + 1]) / 2.0)
            best_split_ent = np.inf
            slen = len(split_list)

            # 求最佳划分点
            best_split = 0.0
            for j in range(slen):
                value = split_list[j]
                new_ent = 0.0
                sub_data_0 = split_continue_data(data, i, value, 0)
                sub_data_1 = split_continue_data(data, i, value, 1)
                prob0 = len(sub_data_0) / len(data)  # 概率
                new_ent += prob0 * cal_ent(sub_data_0)
                prob1 = len(sub_data_1) / len(data)
                new_ent += prob1 * cal_ent(sub_data_1)
                if new_ent < best_split_ent:
                    best_split_ent = new_ent
                    best_split = j  # 最佳划分点
            best_split_dic[feat_list[i]] = split_list[best_split]
            info_gain = base_ent - best_split_ent

            info_gain_dic[feat_list[i]] = info_gain

        # 离散特征
        else:
            new_ent = 0.0
            for value in unique_vals:
                sub_data = split_data(data, i, value)
                prob = len(sub_data) / len(data)
                new_ent += prob * cal_ent(sub_data)
            info_gain = base_ent - new_ent

            info_gain_dic[feat_list[i]] = info_gain

        if info_gain > best_info_gian:
            # noinspection PyUnusedLocal
            best_info_gian = info_gain
            best_feat = i

    # 对连续特征进行二值化处理！！！
    if type(data[0][best_feat]).__name__ == ('float' or 'int'):
        best_split_value = best_split_dic[feat_list[best_feat]]
        feat_list[best_feat] = feat_list[best_feat] + '<=' + str(best_split_value)
        for i in range(np.shape(data)[0]):
            if data[i][best_feat] <= best_split_value:
                data[i][best_feat] = 1  # 此处改变了data的数据
            else:
                data[i][best_feat] = 0  # 此处改变了data的数据

    # sorted_list = sorted(info_gain_dic.items(), key=lambda item: item[1], reverse=True)
    # print(info_gain_dic)
    # print(sorted_list[0])

    return best_feat


# 若特征已划分完而节点下的样本还没有统一的类，则投票
#
def major_cnt(class_list):
    """
    :param class_list: 类别列表
    :return: 多数类
    """
    class_count = {}

    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1

    return max(class_count)


# 产生决策树
#
def creat_tree(data, feat_list):
    """
    :param data: 数据集
    :param feat_list: 特征列表
    :return: 字典树
    """
    class_list = [example[-1] for example in data]

    # 若类别一致则不划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 若属性集为空，返回所含样本最多的类别
    if len(data[0]) == 1:
        return major_cnt(class_list)

    best_feat = choose_bestfeature_to_split(data, feat_list)
    best_feat_name = feat_list[best_feat]
    my_tree = {best_feat_name: {}}
    del (feat_list[best_feat])

    feat_values = [example[best_feat] for example in data]
    unique_vals = set(feat_values)  # 包含所有不同的特征值
    for value in unique_vals:
        sub_feat_list = feat_list[:]  # python参数按引用传递
        my_tree[best_feat_name][value] = creat_tree(split_data(data, best_feat, value), sub_feat_list)

    return my_tree


xg_Data = pd.read_excel('C:/Users/sumlo/Documents/Python_Study/xg_41.xlsx', index_col=0)
dataset = xg_Data.values.tolist()
feat_list_set = xg_Data.columns.values.tolist()

my_Tree = creat_tree(dataset, feat_list_set)
print(my_Tree)

pickle_store.store(my_Tree, 'id3_tree.txt')
