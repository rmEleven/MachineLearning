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

def feature_mapping(x, y, power, as_ndarray=False):
    '''
    将二维特征x和y进行多项式映射
    power用于指定多项式特征的最高次数
    
    例如：
    input:  x = 2, y = 3, power = 2
    return: 1 2 3 4 6 9
    '''
    data = {"f{}{}".format(i - p, p): np.power(x, i - p) * np.power(y, p)
                for i in np.arange(power + 1)
                for p in np.arange(i + 1)
            }

    if as_ndarray:
        return pd.DataFrame(data).values
    else:
        return pd.DataFrame(data)
    
def regularized_cost(theta, X, y, lr=1):
    theta_1_to_n = theta[1:]
    regularized_term = (lr / (2 * len(X))) * np.power(theta_1_to_n, 2).sum()
    return cost(theta, X, y) + regularized_term

def regularized_gradient(theta, X, y, lr=1):
    theta_1_to_n = theta[1:]
    regularized_theta = (lr / len(X)) * theta_1_to_n
    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    return gradient(theta, X, y) + regularized_term

