import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn import preprocessing


def nan_processing(df):
    nulls = df.columns[df.isna().any()]
    for i in nulls:
        print(f"In the column {i}, we have {df[i].isna().sum()} NaN's")
    df.fillna(axis=0, inplace=True)
    print(df.isna().sum().sum(), 'NaN values remained after drop')
    return df


def visual(train):
    pass


if __name__ == '__main__':

    path = r'C:/Users/tzach/PycharmProjects/Healthcare analytics/healthcare'
    train = pd.read_csv(path + r'/train_data.csv')
    test = pd.read_csv(path + r'/test_data.csv')
    df = pd.concat(train, test)
    #Pre-processing
    df.drop(['case_id'], axis=1, inplace=True)
    df = nan_processing(df).reset_index(drop=True)   #NaN
    #label = train.Stay  #label
    LE = preprocessing.LabelEncoder()
    df = LE.fit_transform(y)[source]

    for i in train.columns:
        if type(i) != float or type(i) != int:
            print(i)
            #sns.barplot(x=i,y= hue='Stay', data=train)
    #corrmat = train.corr()
    #sns.heatmap(corrmat, square=True, annot=True)
    #sns.jointplot(x='Age', y='Stay', data=train)


