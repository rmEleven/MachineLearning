import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
sns.set(context='notebook', style='ticks', font_scale=1.5)

from sklearn.metrics import classification_report
import scipy.optimize as opt

from functions import *


'''读取数据'''
data = pd.read_csv('data1.txt', header=None, names=['exam1', 'exam2', 'admitted'])
# print(data.head())      # 显示前5条数据
# print(data.describe())  # 显示统计信息


'''数据可视化'''
positive = data[data['admitted'].isin([1])]  # 正向类
negative = data[data['admitted'].isin([0])]  # 负向类

# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', marker='o', label='Admitted')
# ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='x', label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam1 Score')
# ax.set_ylabel('Exam2 Score')
# plt.show()


'''测试sigmoid函数'''
# nums = np.arange(-10, 10, step=1)
# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(nums, sigmoid(nums), 'r')
# ax.set_xlabel('z', fontsize=18)
# ax.set_ylabel('g(z)', fontsize=18)
# ax.set_title('sigmoid function', fontsize=18)
# plt.show()


'''拟合参数'''
X = get_X(data)
y = get_y(data)
theta = np.zeros(3)
# print(X.shape)
# print(y.shape)
# print(theta.shape)
res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)
# print(res)
# print(cost(res.x, X, y))
final_theta = res.x


'''模型验证'''
predictions = predict(final_theta, X)
correct = [1 if(a == b) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct))) % len(correct)
# print('accuracy = {0}%'.format(accuracy))
# print(classification_report(y, predictions))


'''寻找决策边界'''
# theta0 + theta1 * x1 + theta2 * x2 = 0
x1 = np.arange(130, step=0.1)
x2 = -final_theta[0] / final_theta[2] - final_theta[1] / final_theta[2] * x1
# 创建一个带有线性回归线的散点图
# 横轴的变量名 纵轴的变量名 颜色变量 数据集 图形的高度 图片的宽高比 是否绘制线性回归线 散点的属性
sns.lmplot(x='exam1', y='exam2', hue='admitted', data=data, 
           height=6, aspect=1.5, fit_reg=False, scatter_kws={'s': 25})
plt.plot(x1, x2, 'grey')  # 绘制灰色的决策边界
plt.xlim(0, 130)  # 设置x轴的范围
plt.ylim(0, 130)  # 设置y轴的范围
plt.title('Decision Boundary')  # 设置图表的标题
plt.show()  # 显示图表
