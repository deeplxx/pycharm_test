import numpy as np
import pandas as pd
import re


def format_trans_libsvm(dataset, labelset):
    data_list = []
    label_list = []
    for i in range(len(dataset)):
        data_dic = {}
        for j in range(len(dataset[0])):
            data_dic[j+1] = dataset[i][j]

        data_list.append(data_dic)

    for i in range(len(labelset)):
        if labelset[i] == 0:
            label_list.append(-1)
        else:
            label_list.append(1)

    return label_list, data_list


def format_trans_smo(dataset, labelset):
    data_list = [[r[col] for r in dataset] for col in range(len(dataset[0]))]
    # data_list = [map(list, zip(*dataset))]
    label_list = []
    for i in range(len(labelset)):
        if labelset[i] == 0:
            label_list.append(-1)
        else:
            label_list.append(1)

    return data_list, label_list


def format_trans_xg(dataset, labelset):
    label_list = []
    for i in range(len(labelset)):
        if labelset[i] == 0:
            label_list.append(-1)
        else:
            label_list.append(1)

    datamat = np.mat(dataset)
    labelmat = np.mat(label_list).transpose()

    return datamat, labelmat


def file_read(filename):
    fr = open(filename)
    arr_lines = fr.readlines()
    num_lines = len(arr_lines)
    ret_arr = np.tile('aaaaaa', (num_lines, 3))
    label_arr = []

    index = 0
    for line in arr_lines:
        line = line.strip()  # 去除首尾空格
        list_from_line = re.split('[,\s]', line)
        ret_arr[index, :] = list_from_line[1:]
        label_arr.append(list_from_line[-1])
        index += 1

    fr.close()
    ret_df = pd.DataFrame(ret_arr[1:, :], columns=ret_arr[0, :])

    return ret_df
