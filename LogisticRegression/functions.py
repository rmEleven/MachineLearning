import numpy as np
import pandas as pd


def get_X(df):
    ones = pd.DataFrame({'ones': np.ones(len(df))})  # ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)             # 合并数据，根据列合并
    return data.iloc[:, :-1].values                  # 返回ndarray

def get_y(df):
    return np.array(df.iloc[:, -1])  # 返回df的最后一列

def normalize_feature(df):
    return (df - df.mean()) / df.std  # 特征缩放 标准化

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

def gradient(theta, X, y):
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

def predict(theta, X):
    probability = sigmoid(X @ theta)
    return  [1 if x >= 0.5 else 0 for x in probability]