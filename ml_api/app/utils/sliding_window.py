import numpy as np


def sliding_window(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        window = data[i:(i + window_size)]
        target = data[i + window_size]
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)
