import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report


def load_data(path, transpose=True):
    '''加载数据的函数'''
    data = sio.loadmat(path)  # 读取MATLAB格式的数据文件

    y = data.get('y')  # 从加载的数据中获取y变量，它是一个形状为(5000, 1)的二维numpy数组。
    y = y.reshape(y.shape[0])  # 将y变量从二维数组改变为一维数组，使其形状为(5000,)。

    X = data.get('X')  # 从加载的数据中获取X变量，它是一个形状为(5000, 400)的二维numpy数组。

    if transpose:
        # 将每个样本的数据从(400,)的一维数组形式重构为(20, 20)的二维数组形式,然后再转置
        X = np.array([im.reshape((20, 20)).T for im in X])

        # 将每个样本的数据从(20, 20)的二维数组形式重构为(400,)的一维数组形式。
        X = np.array([im.reshape(400) for im in X])

    return X, y  # 返回处理后的特征向量矩阵X和标签向量y。

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def load_weight(path):
    '''加载神经网络训练好的权重数据'''
    data = sio.loadmat(path)  # 读取MATLAB格式的数据文件
    return data['Theta1'], data['Theta2']  # 从读取的数据中取出权重参数


'''加载数据'''

X, y = load_data('data1.mat',transpose=False)  # 加载特征向量矩阵X和标签向量y
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # 向X的第一列插入值为1的常数列（拟合截距项）
theta1, theta2 = load_weight('weights1.mat')  # 加载神经网络训练好的权重

print(X.shape)
print(y.shape)
print(theta1.shape)
print(theta2.shape)


'''前馈预测'''

a1 = X  # (5000, 401)

z2 = a1 @ theta1.T  # (5000, 401) @ (25,401).T = (5000, 25)
z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)  # 向z2的第一列插入值为1的常数列（拟合截距项）(5000, 26)
a2 = sigmoid(z2)

z3 = a2 @ theta2.T  # (5000, 26) @ (10, 26).T = (5000, 10)
a3 = sigmoid(z3)

y_pred = np.argmax(a3, axis=1) + 1  # 选取神经元输出值最大的作为预测结果


'''准确率'''
print(classification_report(y, y_pred))  # 容易过拟合
