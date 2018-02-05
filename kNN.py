import numpy as np


def classify0(in_x, datamat, labelmat, k):
    m = np.shape(datamat)[0]
    diffmat = np.tile(in_x, (m, 1)) - datamat
    diffarr = np.array(diffmat)
    sq_diffarr = diffarr ** 2
    sq_distance = sq_diffarr.sum(axis=1)
    distance = sq_distance ** 0.5
    sorted_dis_indices = distance.argsort()

    class_count = {}
    for i in range(k):
        vote_i_label = labelmat[sorted_dis_indices[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1

    sorted_class_count = sorted(class_count.iteritems(), key=np.operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]
