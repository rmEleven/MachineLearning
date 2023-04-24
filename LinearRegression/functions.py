import pandas as pd
import numpy as np

def get_X(df):
    """
    函数功能:构造特征矩阵X。
    参数:    传入一个数据框df。
    返回值:  返回一个特征矩阵X,其中第一列全为1,其余列为传入数据框df中的数据。
    """
    # 创建一个元素全为1的数据框ones，长度为传入数据框df的行数。
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    # 将ones和传入的数据框df沿着列的方向(axis=1)合并，得到特征矩阵data。
    data = pd.concat([ones, df], axis=1)
    # 返回特征矩阵data中除了最后一列以外的所有列，并转化为矩阵形式。
    return data.iloc[:, :-1].values

def get_y(df):
    """
    函数功能:构造目标值向量y。
    参数:传入一个数据框df。
    返回值:返回一个目标值向量y,其中y等于传入数据框df中的最后一列数据。
    """
    # 返回数据框df的最后一列数据，转化为数组形式。
    return np.array(df.iloc[:, -1])

def normalize_feature(df):
    """
    函数功能:对传入的数据框df进行特征缩放。
    参数:传入一个数据框df。
    返回值:返回特征缩放后的数据框。
    """
    # 对传入的数据框df的每一列进行特征缩放，即对每一列的数据进行标准化，使其均值为0，标准差为1。
    # 使用apply方法对每一列的数据进行缩放，返回缩放后的数据框。
    return df.apply(lambda column: (column - column.mean()) / column.std())

def lr_cost(theta, X, y):
    """
    函数功能:线性回归的代价函数
    参数:
    theta - 参数数组，包含所有参数
    X     - 输入特征矩阵，大小为 m*(n+1),第一列为全为1的向量,表示截距
    y     - 真实输出值，大小为 m*1
    返回:cost - 线性回归的代价值
    """
    m = X.shape[0]  # 样本数
    inner = X @ theta - y  # 假设函数计算结果与实际结果的误差
    square_sum = inner.T @ inner  # 平方和
    cost = square_sum / (2 * m)  # 损失函数
    return cost

def gradient(theta, X, y):
    """
    梯度下降算法，用于求解线性回归模型参数
    :param theta: 参数向量，大小为 (n+1, 1)，其中 n 是特征数目
    :param X: 特征矩阵，大小为 (m, n+1)，其中 m 是样本数目
    :param y: 标签向量，大小为 (m, 1)
    :return: 梯度向量，大小与 theta 相同
    """
    m = X.shape[0]                 # 样本数目
    inner = X.T @ (X @ theta - y)  # 内部计算，大小为 (n+1, 1)
    return inner / m               # 返回梯度，大小与 theta 相同

def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
    """
    批量梯度下降函数。拟合线性回归，返回参数和代价。
    """
    # 初始化代价数据列表，将起始参数theta下的代价添加进去。
    cost_data = [lr_cost(theta, X, y)]
    # 拷贝一份参数theta，不影响原有的theta。
    _theta = theta.copy()
    # 对于给定的epoch轮数，执行下面的梯度下降操作。
    for _ in range(epoch):
        # 计算梯度下降的更新量，将更新量乘以学习率alpha之后，从_theta中减去，得到新的_theta。
        _theta = _theta - alpha * gradient(_theta, X, y)
        # 将新参数_theta下的代价添加到代价数据列表中。
        cost_data.append(lr_cost(_theta, X, y))
    # 返回最终权重参数_theta以及每轮迭代下的代价数据。
    return _theta, cost_data

def normalEqn(X, y):
    """
    """
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta