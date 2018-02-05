import numpy as np


# sigmoid
#
def sigmoid(x, derivative=False):
    y = 1 / (1 + np.exp(-x))
    if derivative:
        return y
    else:
        return y * (1 - y)


# 神经元类
#
class Neuron:
    def __init__(self, len_input):
        self.weights = np.random.random(len_input) * 0.1  # 初始化权值
        self.vars = np.random.random(len_input) * 0.1
        self.input = np.ones(len_input)  # 输入
        self.output1 = np.ones(len_input)  # 隐层输出
        self.output2 = np.ones(len_input)  # 输出
        self.delta_item = 0  # 误差项
        self.last_weights_add = 0  # 最后一次权值增量

    # 计算输出
    def calc_output(self, input_):
        self.input = input_
        self.output1 = sigmoid(np.dot(self.weights, self.input.T))
        self.output2 = sigmoid(np.dot(self.vars, self.output1.T))

        return self.output1, self.output2

    # 获取反馈差值
    def get_back_weight(self):

        return self.weights * self.delta_item

    # 更新权值
    def update_weights(self, label=0, back_weight=0, alpha=0.1, layer='output'):
        if layer == 'output':
            self.delta_item = (label - self.output2) * sigmoid(self.output2, derivative=True)
        elif layer == 'hidden':
            self.delta_item = back_weight * sigmoid(self.output)

        weight_add = self.input * self.delta_item * alpha + 0.9 * self.last_weights_add

        self.weights += weight_add
        self.last_weights_add = weight_add
