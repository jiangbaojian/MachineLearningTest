# -*- coding:utf-8 -*-
#__author__: 'Baojian Jiang'
#__time__:'2018/12/3'
# https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy/notebook
# 学习的这个
# from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

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
    # print(age_exist)
    age_x = age_exist.values[:, 1:]
    age_y = age_exist.values[:, 0]
    rfr = RandomForestRegressor(n_estimators=1000)
    rfr.fit(age_x, age_y)
    age_hat = rfr.predict(age_null.values[:, 1:])
    data1.loc[(data1.Age.isnull()), 'Age'] = age_hat
    # print(age_hat)
    label = LabelEncoder()
    for dataset in data_cleaner:
        dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
        dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    # print(data1.isnull().sum())
    # embarked_data = pd.get_dummies(data1.Embarked)
    # print(embarked_data)
    target = ['Survived']
    data1_x = ['Sex', 'Pclass', 'Embarked', 'Parch', 'Age', 'Fare']
    # data1_x_calc = ['Embarked_Code']
    data1_xy = target + data1_x
    print('Original x y', data1_xy)
    data1_dummy = pd.get_dummies(data1[data1_x]) #转换为one-hot编码
    x = data1_dummy.columns.tolist()
    # print(x)
    # print(data_train_original.head())
    # print(data1_dummy.head())
    # data1_dummy.to_csv('data/new.csv')
    data1_y = data1['Survived']
    return data1_dummy, data1_y

def show_accuracy(a, b, tip):
    acc = a.ravel() ==b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    print('%s正确率:%.3f ' % (tip, acc_rate))
    return acc_rate

if __name__ == '__main__':
    # pd.set_option('display.max_columns', 1000)
    # pd.set_option('display.max_colwidth', 300)
    train_data_file_path = 'data/train.csv'
    test_data_file_path = 'data/test.csv'
    x, y = load_data(train_data_file_path, test_data_file_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train)
    y_lr_hat = lr.predict(x_test)
    lr_acc = accuracy_score(y_test, y_lr_hat)
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(x_train, y_train)
    y_rfc_hat = rfc.predict(x_test)
    rfc_acc = accuracy_score(y_test, y_rfc_hat)
    # XGB
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}
    bst = xgb.train(param, data_train, num_boost_round=100, evals=watch_list)
    y_hat = bst.predict(data_test)
    y_hat[y_hat > 0.5] = 1
    y_hat[~(y_hat > 0.5)] = 0
    xgb_acc = accuracy_score(y_test, y_hat)
    print('Logistic回归：%.3f%%' % lr_acc)
    print('随机森林：%.3f%%' % rfc_acc)
    print('XGBoost：%.3f%%' % xgb_acc)


