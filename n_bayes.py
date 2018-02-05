import numpy as np


# 将要检测的文本转换为文本向量(词集模型   )
#
def set_of_words_vec(vocab_list, input_set):
    """
    :param vocab_list: 训练的库样本
    :param input_set: 测试样本
    :return: 测试样本转化的向量
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)

    return return_vec


# 词袋模型
#
def bag_of_words_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
        else:
            print("the word: %s is not in my vocabulary!" % word)

    return return_vec


# 训练函数
#
def train_nb(train_vec, train_label):
    num_traindocs = len(train_vec)
    num_words = len(train_vec[0])
    p_abusive = (sum(train_label) + 1) / (float(num_traindocs) + 2)  # 拉普拉斯修正,表示p(c)
    p0_num = np.ones(num_words)  # 。。。D(c，x) + 1
    p1_num = np.ones(num_words)  # 。。。
    p0_denom = 2.0  # 。。。D(c) + 2
    p1_denom = 2.0  # 。。。
    for i in range(train_vec):
        if train_label == 1:
            p1_num += train_vec[i]
            p1_denom += sum(train_vec[i])
        else:
            p0_num += train_vec[i]
            p0_denom += sum(train_vec[i])

    p1_vec = np.log(p1_num / p1_denom)  # 防止数值太小四舍五入变成0
    p0_vec = np.log(p0_num / p0_denom)  # 后续运算累乘变成累加！

    return p0_vec, p1_vec, np.log(p_abusive)


# 分类函数
#
def classify_nb(vec_classify, p0_vec, p1_vec, p_abusive):
    """
    :param vec_classify: 待分类向量
    :param p0_vec: 训练结果
    :param p1_vec: 训练结果
    :param p_abusive: 训练结果
    :return: 分类结果
    """
    p1 = sum(vec_classify * p1_vec) + p_abusive  # p(w|x) * p(c)
    p0 = sum(vec_classify * p0_vec) + (1 - p_abusive)

    if p1 > p0:
        return 1
    else:
        return 0
