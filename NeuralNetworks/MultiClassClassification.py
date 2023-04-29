'''使用逻辑回归来识别手写数字'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

from scipy.optimize import minimize


def load_data(path, transpose=True):
    '''加载数据的函数'''
    data = sio.loadmat(path)
    y = data.get('y')  # shape为二位数组 (5000,1)
    X = data.get('X')  # shape为二位数组 (5000,400)

    if transpose:
        # 对于这个数据集，需要把每一组数据（图像）转置（因此这里先reshape为20*20，然后再转置）
        X = np.array([im.reshape((20, 20)).T for im in X])

        # 把每一个sample reshape为vector
        X = np.array([im.reshape(400) for im in X])

    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg

def gradient_with_loop(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])
    
    return grad

def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y
    
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    
    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    
    return np.array(grad).ravel()

def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]
    
    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))
    
    # 插入一列全为1的向量，对应的是theta0
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    
    # 对每一个类别计算最优化参数theta
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        
        # 最小化目标函数
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i-1,:] = fmin.x
    
    return all_theta

def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]
    
    # 插入一列全为1的向量，对应的是theta0
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    
    # 计算每一类的概率
    h = sigmoid(X * all_theta.T)
    
    # 找到最大概率对应的类别
    h_argmax = np.argmax(h, axis=1)
    
    # 加1是因为需要还原为从1开始索引的数字
    h_argmax = h_argmax + 1
    
    return h_argmax


raw_X, raw_y = load_data('data1.mat')
print(raw_X.shape)  # (5000, 400) 20*20灰度像素
print(raw_y.shape)  # (5000, 1)

all_theta = one_vs_all(raw_X, raw_y, 10, 1)

y_pred = predict_all(raw_X, all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, raw_y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))
