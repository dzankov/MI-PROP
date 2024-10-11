
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale_descriptors(x_train, x_test):
    scaler = MinMaxScaler()
    scaler.fit(np.vstack(x_train))

    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()
    for i, bag in enumerate(x_train):
        x_train_scaled[i] = scaler.transform(bag)
    for i, bag in enumerate(x_test):
        x_test_scaled[i] = scaler.transform(bag)

    return x_train_scaled, x_test_scaled
