
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame


def df_to_list_of_bags(df_descr):
    list_of_bags = []
    for i in df_descr.index.unique():
        x = df_descr.loc[i:i].values
        list_of_bags.append(x)

    return list_of_bags
