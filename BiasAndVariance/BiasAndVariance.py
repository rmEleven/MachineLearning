import numpy as np
import scipy.optimize as opt

import pandas as pd
import scipy.io as sio

import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """使用 np.ravel 把一列数据 flattern 为 vector"""
    d = sio.loadmat('data.mat')  # 加载数据文件
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])  # 扁平化处理

def cost(theta, X, y):
    """计算损失"""
    m = X.shape[0]                # 获取样本数 m
    inner = X @ theta - y         # 计算假设函数的输出 h(x) 与 y 的偏差
    square_sum = inner.T @ inner  # 计算残差平方和
    cost = square_sum / (2 * m)   # 计算损失函数的值
    return cost                   # 返回损失函数的值

def regularized_cost(theta, X, y, l=1):
    """计算正则化损失"""
    m = X.shape[0]                                         # 获取样本数 m
    inner = X @ theta - y                                  # 计算假设函数的输出 h(x) 与 y 的偏差
    cost = (inner.T @ inner) / (2 * m)                     # 计算损失函数的值
    regularized_term = np.sum(theta[1:]**2) * l / (2 * m)  # 计算正则化项的值
    return cost + regularized_term                         # 返回带正则化项的损失函数的值

def gradient(theta, X, y):
    '''计算梯度'''
    m = X.shape[0]                 # 获取样本数量 m
    inner = X.T @ (X @ theta - y)  # 计算梯度 (m,n).T @ (m, 1) -> (n, 1)
    return inner / m               # 对梯度除以样本数量 m，得到平均梯度

def regularized_gradient(theta, X, y, l=1):
    '''计算正则化梯度'''
    m = X.shape[0]                                   # 获取样本数量
    regularized_term = theta.copy()                  # 复制 theta，保证 theta 不被修改
    regularized_term[0] = 0                          # 不对截距 theta0 进行正则化
    regularized_term = (l / m) * regularized_term    # 计算正则化项
    return gradient(theta, X, y) + regularized_term  # 返回带正则化项的梯度

def linear_regression_np(X, y, l=1):
    '''线性回归训练'''
    # init theta
    theta = np.ones(X.shape[1])
    # train it
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': False})
    return res

def normalize_feature(df):
    '''数据标准化'''
    # 对每列进行标准化处理，公式为 (x - mean) / std
    return df.apply(lambda column: (column - column.mean()) / column.std())

def poly_features(x, power, as_ndarray=False):
    """将输入的x按幂次扩展，并将结果转换为DataFrame或numpy数组形式"""
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}  # 构建一个幂次为power的字典
    df = pd.DataFrame(data)                 # 将字典转化为DataFrame
    return df.values if as_ndarray else df  # 根据需要将DataFrame转化为numpy数组或保持原始形式

def prepare_poly_data(*args, power):
    """幂次扩展、标准化、添加偏置项"""
    def prepare(x):
        df = poly_features(x, power=power)    # expand feature
        ndarr = normalize_feature(df).values  # normalization
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)  # add intercept term
    return [prepare(x) for x in args]

def plot_learning_curve(X, y, Xval, yval, l=0, re_cost=False):
    '''绘制学习曲线'''

    training_cost, cv_cost = [], []  # 初始化训练集和交叉验证集的代价
    m = X.shape[0]                   # 获取样本数量

    for i in range(1, m + 1):  # 遍历样本
        res = linear_regression_np(X[:i, :], y[:i], l=l)  # 使用线性回归求解参数，并进行正则化

        tc = regularized_cost(res.x, X[:i, :], y[:i], l=l) if re_cost else cost(res.x, X[:i, :], y[:i])  # 计算训练集代价
        cv = regularized_cost(res.x, Xval, yval, l=l) if re_cost else cost(res.x, Xval, yval)            # 计算交叉验证集代价

        training_cost.append(tc)  # 将训练集代价加入数组
        cv_cost.append(cv)        # 将交叉验证集代价加入数组

    # 绘制学习曲线
    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')     # 训练集代价
    plt.plot(np.arange(1, m + 1), cv_cost, label='cross-validataion cost')  # 交叉验证集代价
    plt.legend(loc=1)  # 添加图例


'''读取数据'''
X, y, Xval, yval, Xtest, ytest = load_data()

# df = pd.DataFrame({'water_level':X, 'flow':y})  # 将 X 和 y 转换成 pandas 的 DataFrame 类型
# # 生成一个散点图和一条线性回归线，其中 x 轴为 water_level，y 轴为 flow，数据来源于 df，且不进行线性回归拟合（fit_reg=False），高度为 7
# sns.lmplot(x='water_level', y='flow', data=df, fit_reg=False, height=7)
# plt.show()  # 显示绘制的图形


'''拟合数据'''
# 将 X、Xval 和 Xtest 三个一维数组转换为二维数组，并在其最左侧插入一列 1，用于计算线性回归模型的偏置。
X_, Xval_, Xtest_ = X.copy() , Xval.copy() , Xtest.copy()  # 备份一维数组
X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]

final_theta = linear_regression_np(X, y, l=0).get('x')
b = final_theta[0] # intercept
k = final_theta[1] # slope

# plt.scatter(X[:, 1], y, label="Training data")          # 训练数据点
# plt.plot(X[:, 1], X[:, 1] * k + b, label="Prediction")  # 拟合的线性回归模型
# plt.legend(loc=2)  # 添加图例
# plt.show()         # 显示绘制的图形


'''绘制学习曲线'''
plot_learning_curve(X, y, Xval, yval, l=0, re_cost=True)
plt.show()  # 欠拟合


'''多项式回归数据'''
X_poly, Xval_poly, Xtest_poly = prepare_poly_data(X_, Xval_, Xtest_, power=8)

# plot_learning_curve(X_poly, y, Xval_poly, yval, l=0)
# plt.show()  # 过拟合
# plot_learning_curve(X_poly, y, Xval_poly, yval, l=1)
# plt.show()  # 过拟合
# plot_learning_curve(X_poly, y, Xval_poly, yval, l=100)
# plt.show()  # 欠拟合


