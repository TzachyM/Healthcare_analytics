import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt


def load_data():
    path = r'C:/Users/tzach/PycharmProjects/Healthcare analytics/healthcare'
    train = pd.read_csv(path + r'/train_data.csv')
    test = pd.read_csv(path + r'/test_data.csv')
    return train, test


def preprocessing(train, test):
    train = nan_processing("train", train).reset_index(drop=True)     # NaN (test and train came with NaN's)
    stay_value_dict = {'0-10': 0, '11-20': 1, '21-30': 2, '31-40': 3, '41-50': 4, '51-60': 5, '61-70': 6, '71-80': 7,
                      '81-90': 8, '91-100': 9, 'More than 100 Days': 10}
    train.Stay = train.Stay.map(stay_value_dict)
    test = nan_processing("test", test).reset_index(drop=True)
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    df.drop(['case_id', 'patientid'], axis=1, inplace=True)

    age_value_dict = {'0-10': 0, '11-20': 1, '21-30': 2, '31-40': 3, '41-50': 4, '51-60': 5, '61-70': 6, '71-80': 7,
                      '81-90': 8, '91-100': 9}
    df.Age = df.Age.map(age_value_dict)
    illness_value_dict = {'Minor': 0, 'Moderate': 1, 'Extreme': 2}
    df['Severity of Illness'] = df['Severity of Illness'].map(illness_value_dict)
    visual(df.iloc[:len(train), :])
    df = pd.get_dummies(df)
    train_df = df.iloc[:len(train), :].reset_index(drop=True)
    test_df = df.iloc[len(train):, :].reset_index(drop=True)
    test_df.drop(['Stay'], axis=1, inplace=True)


    return train_df, test_df

def nan_processing(data, df):
    print(data, ':')
    nulls = df.columns[df.isna().any()]
    for i in nulls:
        print(f"In the column {i}, we have {df[i].isna().sum()} NaN's")
        df[i].fillna(df[i].median(), inplace=True)
    print(df.isna().sum().sum(), 'NaN values remained after processing')
    return df


def visual(train_df):
    corrmat = train_df.corr()
    sns.heatmap(corrmat, square=True, annot=True)
    sns.catplot(x='Visitors with Patient', y='Stay', kind="bar", data=train_df)


if __name__ == '__main__':


    train, test = load_data()
    train_df, test_df = preprocessing(train, test)



    for i in train_df.select_dtypes(exclude=['object']):
        print(i)


