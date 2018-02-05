import matplotlib.pyplot as plt

# 定义文本框和箭头格式
#
decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


# 绘制带箭头的注解
#
def creat_plot():
    fig = plt.figure(1, facecolor='w')
    fig.clf()  # 清除fig中的内容
    creat_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node('juece', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('ye', (0.8, 0.1), (0.3, 0.8), leaf_node)

    plt.show()


def plot_node(node_txt, center_pt, parent_pt, node_type):
    # noinspection PyUnresolvedReferences
    creat_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                            xytext=center_pt, textcoords='axes fraction',
                            va='center', ha='center', bbox=node_type, arrowprops=arrow_args)
