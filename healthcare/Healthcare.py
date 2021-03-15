import pandas as pd
import numpy as np


def visual(train):
    pass

if __name__ == '__main__':
    path = r'C:/Users/tzach/PycharmProjects/Healthcare analytics/healthcare/train_data.csv'
    train = pd.read_csv(path)

    nulls = train.columns[train.isna().any()]
    for i in nulls:
        print(f"In the columns {i}, we have {train[i].isna().sum()} NaN's")
a