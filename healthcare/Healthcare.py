import pandas as pd
import numpy as np
import seaborn as sns

def nan_processing(train):
    nulls = train.columns[train.isna().any()]
    for i in nulls:
        print(f"In the columns {i}, we have {train[i].isna().sum()} NaN's")
    train.dropna(axis=0, inplace=True)
    print(train.isna().sum().sum(), 'NaN values remained after drop')
    return train
def visual(train):
    pass

if __name__ == '__main__':
    path = r'C:/Users/tzach/PycharmProjects/Healthcare analytics/healthcare/train_data.csv'
    train = pd.read_csv(path)

    #Pre-processing
    train = nan_processing(train)   #NaN
    label = train.Stay  #label

