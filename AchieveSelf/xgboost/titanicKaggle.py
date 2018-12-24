# -*- coding:utf-8 -*-
#__author__: 'Baojian Jiang'
#__time__:'2018/12/3'
# https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy/notebook
# 学习的这个
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def load_data(train_file_name, test_file_name):
    data_train_original = pd.read_csv(train_file_name)
    data_test_original = pd.read_csv(test_file_name)
    data1 = data_train_original.copy(deep=True)
    data_cleaner = [data1, data_test_original]
    # print(data_train_original.sample(10))
    # print(data_test_original.head())
    # print('data.describe()= \n', data_train_original.describe())
    drop_columns = ['PassengerId', 'Cabin', 'Ticket']
    data1.drop(drop_columns, axis=1, inplace=True)
    print(data1.isnull().sum())
    data1['Sex'] = data1['Sex'].map({'female': 0, 'male': 1}).astype(int)
    print('使用随机森林预测年龄')
    data_for_age = data1[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    age_exist = data_for_age.loc[(data1.Age.notnull())] #年龄不缺失数据
    age_null = data_for_age.loc[(data1.Age.isnull())]
    print(age_exist)
    age_x = age_exist.values[:, 1:]
    age_y = age_exist.values[:, 0]
    rfr = RandomForestRegressor(n_estimators=1000)
    rfr.fit(age_x, age_y)
    age_hat = rfr.predict(age_null.values[:, 1:])
    print(age_hat)
    # print(data1)
    # label = LabelEncoder()
    # for dataset in data_cleaner:
    #     # print(dataset)
    #     dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    # print(data_train_original.head())


if __name__ == '__main__':
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_colwidth', 300)
    train_data_file_path = 'data/train.csv'
    test_data_file_path = 'data/test.csv'
    load_data(train_data_file_path, test_data_file_path)

