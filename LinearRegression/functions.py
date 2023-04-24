import pandas as pd
import numpy as np

def get_X(df):
    """
    构造特征矩阵X，其中第一列全为1，其余列为传入数据框df中的数据。
    """
    # 创建元素全为1的数据框ones，长度为df的行数
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    # 将ones和df在列的方向上合并，得到特征矩阵data
    data = pd.concat([ones, df], axis=1)
    # 返回除最后一列外的所有列并转化为矩阵
    return data.iloc[:, :-1].values

def get_y(df):
    """
    构造目标值向量y.
    """
    # 返回数据框df的最后一列数据, 转化为数组形式。
    return df.iloc[:, -1].values

def normalize_feature(df):
    """
    对传入的数据框进行特征缩放。
    """
    # 对每一列数据进行标准化，使均值为0，标准差为1
    scaled_df = (df - df.mean()) / df.std()
    return scaled_df

def lr_cost(theta, X, y):
    """
    计算线性回归的代价函数值。
    """
    m = X.shape[0]                # 样本数
    inner = X @ theta - y         # 计算假设函数计算结果与实际结果的误差
    square_sum = inner.T @ inner  # 平方和
    cost = square_sum / (2 * m)   # 代价函数
    return cost

def gradient(theta, X, y):
    """
    梯度下降算法，用于求解线性回归模型参数
    :param theta: 参数向量，大小为 (n+1, 1)，其中 n 是特征数目
    :param X: 特征矩阵，大小为 (m, n+1)，其中 m 是样本数目
    :param y: 标签向量，大小为 (m, 1)
    :return: 梯度向量，大小与 theta 相同
    """
    m = X.shape[0]                      # 样本数目
    inner = X.T @ (X @ theta - y)       # 内部计算，大小为 (n+1, 1)
    return inner / m                    # 返回梯度，大小与 theta 相同

def batch_gradient_descent(theta, X, y, epoch, alpha=0.01):
    """
    批量梯度下降函数。拟合线性回归，返回参数和代价。
    :param theta: 参数向量，大小为 (n+1, 1)，其中 n 是特征数目
    :param X: 特征矩阵，大小为 (m, n+1)，其中 m 是样本数目
    :param y: 标签向量，大小为 (m, 1)
    :param epoch: 迭代次数
    :param alpha: 学习率
    :return: 返回参数和代价
    """
    _theta = theta.copy()               # 拷贝一份参数theta，不影响原有的theta。
    cost_data = [lr_cost(theta, X, y)]  # 初始化代价数据列表，将起始参数theta下的代价添加进去。
    for _ in range(epoch):
        # 更新_theta
        _theta = _theta - alpha * gradient(_theta, X, y)
        # 将新参数_theta下的代价添加到代价数据列表中。
        cost_data.append(lr_cost(_theta, X, y))
    # 返回最终权重参数theta以及每轮迭代下的代价数据。
    return _theta, cost_data

def normalEqn(X, y):
    """
    正规方程函数，用于求解线性回归模型参数。
    """
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta