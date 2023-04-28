import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report


def load_data(path, transpose=True):
    '''加载数据的函数'''
    data = sio.loadmat(path)
    y = data.get('y')  # shape为二位数组 (5000,1)
    y = y.reshape(y.shape[0])  # reshape 为 column vector

    X = data.get('X')  # shape为二位数组 (5000,400)

    if transpose:
        # 对于这个数据集，需要把每一组数据（图像）转置（因此这里先reshape为20*20，然后再转置）
        X = np.array([im.reshape((20, 20)).T for im in X])

        # 把每一个sample reshape为vector
        X = np.array([im.reshape(400) for im in X])

    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']


'''加载数据'''

X, y = load_data('data1.mat',transpose=False)
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # intercept
theta1, theta2 = load_weight('weights1.mat')

print(X.shape)
print(y.shape)
print(theta1.shape)
print(theta2.shape)


'''前馈预测'''

a1 = X

z2 = a1 @ theta1.T # (5000, 401) @ (25,401).T = (5000, 25)
z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)
a2 = sigmoid(z2)

z3 = a2 @ theta2.T
a3 = sigmoid(z3)

y_pred = np.argmax(a3, axis=1) + 1  # numpy is 0 base index, +1 for matlab convention，返回沿轴axis最大值的索引，axis=1代表行


'''准确率'''
print(classification_report(y, y_pred))  # 容易过拟合
