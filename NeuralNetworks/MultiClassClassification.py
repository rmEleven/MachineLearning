'''使用逻辑回归来识别手写数字'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

from scipy.optimize import minimize


def load_data(path, transpose=True):
    '''
    加载数据的函数
    '''
    data = sio.loadmat(path)   # 读取MATLAB格式的数据文件

    X = data.get('X')          # 从加载的数据中获取X变量，它是一个形状为(5000, 400)的二维numpy数组。
    y = data.get('y')          # 从加载的数据中获取y变量，它是一个形状为(5000, 1)的二维numpy数组。

    if transpose:
        '''将X变量转置 转置前后形状不变 仍是(400,)的一维数组'''
        # 将每个样本的数据从(400,)的一维数组形式重构为(20, 20)的二维数组形式,然后再转置
        X = np.array([im.reshape((20, 20)).T for im in X])
        # 将每个样本的数据从(20, 20)的二维数组形式重构为(400,)的一维数组形式。
        X = np.array([im.reshape(400) for im in X])

    return X, y  # 返回处理后的特征向量矩阵X和标签向量y。

def sigmoid(z):
    '''
    逻辑函数
    '''
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y, learningRate):
    '''
    计算正则化代价函数
    '''
    theta = np.matrix(theta)  # 将theta转换为矩阵形式
    X = np.matrix(X)          # 将X转换为矩阵形式
    y = np.matrix(y)          # 将y转换为矩阵形式

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))            # 计算代价函数第一项: -ylogh
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))  # 计算代价函数第二项: (1-y)log(1-h)
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))  # 计算正则化项
    return np.sum(first - second) / len(X) + reg                     # 计算代价函数

def gradient(theta, X, y, learningRate):
    '''
    计算正则化梯度
    '''
    theta = np.matrix(theta)  # 将theta转换为矩阵形式
    X = np.matrix(X)          # 将X转换为矩阵形式
    y = np.matrix(y)          # 将y转换为矩阵形式
    
    # parameters = int(theta.ravel().shape[1])

    error = sigmoid(X * theta.T) - y  # 计算损失
    
    # 计算正则化梯度
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    
    return np.array(grad).ravel()  # 按照行优先的顺序将多维数组转换为一维数组

def one_vs_all(X, y, num_labels, learning_rate):
    '''
    训练一对一全分类分类器的参数
    '''
    rows = X.shape[0]    # 样本数
    params = X.shape[1]  # 特征数

    all_theta = np.zeros((num_labels, params + 1))     # (分类器个数*特征数量)
    X = np.insert(X, 0, values=np.ones(rows), axis=1)  # 插入一列全为1的向量，对应的是theta0，表示偏置项
    
    '''对每一个类别计算最优化参数theta'''
    for i in range(1, num_labels + 1):    # 枚举每个分类器
        theta = np.zeros(params + 1)      # 初始化theta参数
        y_i = np.array([1 if label == i else 0 for label in y])  # 将标签转化为0或1的数组，其中y_i[i] = 1表示样本属于第i类，反之为0
        y_i = np.reshape(y_i, (rows, 1))  # 将标签数组转化为列向量
        
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)  # 最小化目标函数
        all_theta[i-1,:] = fmin.x         # 存储最优参数
    
    return all_theta  # 返回所有分类器的参数矩阵all_theta

def predict_all(X, all_theta):
    '''
    使用训练完毕的分类器预测每个图像的标签
    计算每个类的类概率
    输出具有最高概率的类
    '''
    rows = X.shape[0]   # 数据集样本数
    X = np.insert(X, 0, values=np.ones(rows), axis=1)  # 插入一列全为1的向量，对应的是theta0，表示偏置项
    X = np.matrix(X)                  # 转换为矩阵类型
    all_theta = np.matrix(all_theta)  # 转换为矩阵类型
    
    h = sigmoid(X * all_theta.T)         # 计算每一类的概率
    h_argmax = np.argmax(h, axis=1) + 1  # 找到最大概率对应的类别
    
    return h_argmax  # 返回预测的标签


'''加载数据'''

raw_X, raw_y = load_data('data1.mat')
print(raw_X.shape)  # (5000, 400) 20*20灰度像素
print(raw_y.shape)  # (5000, 1)


'''拟合参数'''

all_theta = one_vs_all(raw_X, raw_y, 10, 1)


'''评估模型'''

y_pred = predict_all(raw_X, all_theta)                           # 用模型对输入参数进行预测
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, raw_y)]  # 与标签进行对比
accuracy = (sum(map(int, correct)) / float(len(correct)))        # 计算准确率
print ('accuracy = {0}%'.format(accuracy * 100))                 # 输出准确率
