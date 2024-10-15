
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame


def df_to_list_of_bags(df_descr):
    list_of_bags = []
    for i in df_descr.index.unique():
        list_of_bags.append(df_descr.loc[i].values)

    return list_of_bags


