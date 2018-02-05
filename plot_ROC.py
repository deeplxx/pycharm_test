import matplotlib.pyplot as plt


def plot_roc(predic_strengths, labelmat):
    cur = (1.0, 1.0)  # 绘制光标的位置
    y_sum = 0.0  # 用于计算auc的值
    num_pos_class = sum(labelmat[labelmat == 1.0])  # 样本中正例的数目
    y_step = 1 / float(num_pos_class)  # y轴上的步长
    x_step = 1 / float(len(labelmat) - num_pos_class)  # x轴上的步长
    sorted_indices = predic_strengths.argsort()

    fig1 = plt.figure()
    fig1.clf()
    ax1 = plt.subplot(111)
    for index in sorted_indices.tolist()[0]:
        if labelmat[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]

        ax1.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
        cur = (cur[0] - del_x, cur[1] - del_y)

    ax1.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for adaboost')
    ax1.axis([0, 1, 0, 1])
    plt.show()

    print('the area under the cur is: ', y_sum * x_step)
