import os
import numpy as np

def load_npys(data_dir, args):
    npys = [np.load(os.path.join(data_dir, i)) for i in args]
    if len(args) == 1:
        return npys[0]
    return npys

def mean_average_precision(Y_pred, Y_true):
    result = 0.0
    for i in range(len(Y_pred)):
        average_precision = precision = 0.0
        true_set = set(Y_true[i])
        for j, item in enumerate(Y_pred[i]):
            if item in true_set:
                precision += 1
                average_precision += precision / (j + 1)

        result += average_precision / len(Y_true[i])
    return result / len(Y_pred)
