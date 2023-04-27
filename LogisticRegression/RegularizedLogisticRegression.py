import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
sns.set(context='notebook', style='ticks', font_scale=1.5)

from sklearn.metrics import classification_report
import scipy.optimize as opt
from sklearn import linear_model

from functions import *


'''读取数据'''
data = pd.read_csv('data2.txt', names=['test1', 'test2', 'accepted'])
# print(data.head())      # 显示前5条数据
# print(data.describe())  # 显示统计信息


'''数据可视化'''
positive = data[data['accepted'].isin([1])]
negative = data[data['accepted'].isin([0])]

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive['test1'], positive['test2'], s=50, c='b', marker='o', label='Accepted')
# ax.scatter(negative['test1'], negative['test2'], s=50, c='r', marker='x', label='Rejected')
# ax.legend()
# ax.set_xlabel('Test 1 Score')
# ax.set_ylabel('Test 2 Score')
# plt.show()


'''拟合参数'''
x1 = np.array(data.test1)
x2 = np.array(data.test2)
X = feature_mapping(x1, x2, power=6, as_ndarray=True)
y = get_y(data)
theta = np.zeros(X.shape[1])
# print(X.shape)
# print(y.shape)
# print(theta.shape)
learningRate = 1
res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y), method='Newton-CG', jac=regularized_gradient)
final_theta = res.x


'''模型验证'''
predictions = predict(final_theta, X)
# print(classification_report(y, predictions))


'''调用线性回归包'''
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X, y.ravel())
# print(model.score(X, y))


def feature_mapped_logistic_regression(power, l):
#     """for drawing purpose only.. not a well generealize logistic regression
#     power: int
#         raise x1, x2 to polynomial power
#     l: int
#         lambda constant for regularization term
#     """
    df = pd.read_csv('data2.txt', names=['test1', 'test2', 'accepted'])
    x1 = np.array(df.test1)
    x2 = np.array(df.test2)
    y = get_y(df)

    X = feature_mapping(x1, x2, power, as_ndarray=True)
    theta = np.zeros(X.shape[1])

    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient)
    final_theta = res.x

    return final_theta

def find_decision_boundary(density, power, theta, threshhold):
    t1 = np.linspace(-1, 1.5, density)
    t2 = np.linspace(-1, 1.5, density)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = feature_mapping(x_cord, y_cord, power)  # this is a dataframe

    inner_product = mapped_cord.values @ theta

    decision = mapped_cord[np.abs(inner_product) < threshhold]

    return decision.f10, decision.f01

def draw_boundary(power, l):
#     """
#     power: polynomial power for mapped feature
#     l: lambda constant
#     """
    density = 1000
    threshhold = 2 * 10**-3

    final_theta = feature_mapped_logistic_regression(power, l)
    x, y = find_decision_boundary(density, power, final_theta, threshhold)

    df = pd.read_csv('data2.txt', names=['test1', 'test2', 'accepted'])
    sns.lmplot(x='test1', y='test2', hue='accepted', data=df, height=6, aspect=1.5, fit_reg=False, scatter_kws={"s": 100})

    plt.scatter(x, y, c='r', s=10)
    plt.title('Decision boundary')
    plt.show()

draw_boundary(power=6, l=1)#lambda=1
draw_boundary(power=6, l=0)  # no regularization, over fitting，#lambda=0,没有正则化，过拟合了
draw_boundary(power=6, l=100)  # underfitting，#lambda=100,欠拟合