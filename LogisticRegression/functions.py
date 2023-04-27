import numpy as np
import pandas as pd

import matplotlib.pyplot as plt  # 用于绘制图形和可视化数据
plt.style.use('fivethirtyeight')  # 设置样式
import seaborn as sns   # 提供更高级别的图形界面和额外的绘图功能
sns.set(context='notebook', style='ticks', font_scale=1.5)  # 绘图环境 绘图的样式 字体的大小

import scipy.optimize as opt


def get_X(df):
    ones = pd.DataFrame({'ones': np.ones(len(df))})  # 创建一个m行1列的DataFrame，元素均为1
    data = pd.concat([ones, df], axis=1)             # 将ones和df按列合并为一个新的DataFrame
    return data.iloc[:, :-1].values                  # 返回合并后DataFrame中除最后一列外的所有列，并将其转换为一个numpy数组对象

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

def feature_mapped_logistic_regression(data, power, lbd):
    '''
    执行特征映射逻辑回归
    '''
    x1 = np.array(data.test1)
    x2 = np.array(data.test2)
    X = feature_mapping(x1, x2, power, as_ndarray=True)
    y = get_y(data)
    theta = np.zeros(X.shape[1])

    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, lbd),
                       method='TNC',
                       jac=regularized_gradient)
    final_theta = res.x

    return final_theta

def find_decision_boundary(density, power, theta, threshhold):
    '''
    在给定的范围内生成一组坐标点，然后将这些坐标点转换为多项式特征映射形式，
    利用事先训练好的参数theta计算每个点的内积，
    然后筛选出内积绝对值小于给定阈值threshold的点作为决策边界。
    '''
    # 创建两个长度为 density 的一维数组,包含从 -1 到 1.5 的均匀分布的数字。
    t1 = np.linspace(-1, 1.5, density)
    t2 = np.linspace(-1, 1.5, density)

    # 使用列表推导式生成坐标对列表 cordinates，包含所有可能的 t1 和 t2 坐标对。
    cordinates = [(x, y) for x in t1 for y in t2]

    # 使用 zip 函数将坐标对列表 cordinates 拆分成两个元组 x_cord 和 y_cord。
    x_cord, y_cord = zip(*cordinates)

    # 获取多项式特征
    mapped_cord = feature_mapping(x_cord, y_cord, power)

    # 计算内积
    inner_product = mapped_cord.values @ theta

    # 选择决策边界
    decision = mapped_cord[np.abs(inner_product) < threshhold]

    # 返回决策边界上的点的横坐标和纵坐标
    return decision.f10, decision.f01

def draw_boundary(data, power, lbd, density=1000, threshhold = 2*10**-3):
    '''
    绘制决策边界与散点图
    '''
    # 执行特征映射逻辑回归
    final_theta = feature_mapped_logistic_regression(data, power, lbd)
    # 选择决策边界
    x, y = find_decision_boundary(density, power, final_theta, threshhold)

    # 创建一个带有决策边界的散点图
    # 横轴的变量名 纵轴的变量名 颜色变量 数据集 图形的高度 图片的宽高比 是否绘制线性回归线 散点的属性
    sns.lmplot(x='test1', y='test2', hue='accepted', data=data, height=6, aspect=1.5, fit_reg=False, scatter_kws={"s": 100})

    plt.scatter(x, y, c='r', s=10)  # 绘制红色的散点决策边界
    plt.title('Decision boundary')  # 设置图表的标题
    plt.show()  # 显示图表
    