import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report


def load_data(path, transpose=True):
    '''
    加载数据的函数
    '''
    data = sio.loadmat(path)   # 读取MATLAB格式的数据文件

    X = data.get('X')          # 从加载的数据中获取X变量，它是一个形状为(5000, 400)的二维numpy数组。
    y = data.get('y')          # 从加载的数据中获取y变量，它是一个形状为(5000, 1)的二维numpy数组。
    y = y.reshape(y.shape[0])  # 将y变量从二维数组改变为一维数组，使其形状为(5000,)。

    if transpose:
        '''将X变量转置 转置前后形状不变 仍是(400,)的一维数组'''
        # 将每个样本的数据从(400,)的一维数组形式重构为(20, 20)的二维数组形式,然后再转置
        X = np.array([im.reshape((20, 20)).T for im in X])
        # 将每个样本的数据从(20, 20)的二维数组形式重构为(400,)的一维数组形式。
        X = np.array([im.reshape(400) for im in X])

    return X, y  # 返回处理后的特征向量矩阵X和标签向量y。

def load_weight(path):
    '''
    加载神经网络训练好的权重数据
    '''
    data = sio.loadmat(path)               # 读取MATLAB格式的数据文件
    return data['Theta1'], data['Theta2']  # 从读取的数据中取出权重参数

def plot_100_image(X):
    '''
    随机从数据集中抽取100张图片进行展示
    X : (5000, 400)
    '''
    size = int(np.sqrt(X.shape[1]))  # 计算图像的边长

    # 从数据集中随机抽取100张图片
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 100 * 400
    # 对抽取的100张图片进行重新排列
    sample_images = X[sample_idx, :]

    # 绘制10x10的网格，将100张图片排列在网格中
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    # 对于每一个网格，绘制一张图片
    for r in range(10):
        for c in range(10):
            # 使用matshow绘制图片
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size)), cmap=matplotlib.cm.binary)
            # 隐藏刻度
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

def plot_hidden_layer(theta):
    '''
    展示神经网络第一层的权重矩阵
    theta: (10285, )
    '''
    final_theta1, _ = deserialize(theta)  # 将一维数组形式的theta还原成二维矩阵形式的theta1
    hidden_layer = final_theta1[:, 1:]    # 获取去掉偏置项后的第一层权重矩阵

    # 创建5*5的画布，用于展示25个隐藏层单元的权重
    fig, ax_array = plt.subplots(nrows=5, ncols=5, sharey=True, sharex=True, figsize=(5, 5))

    # 逐个展示隐藏层单元的权重
    for r in range(5):
        for c in range(5):
            # 将该单元的权重矩阵转换成20*20的矩阵，并绘制为灰度图
            ax_array[r, c].matshow(hidden_layer[5 * r + c].reshape((20, 20)), cmap=matplotlib.cm.binary)
            # 设置坐标轴刻度为空
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

def expand_y(y):
    '''
    将原始的标签y进行扩展，将其转化为一个维度为5000x10的矩阵，
    其中每一行对应一个样本的标签，每行有10个元素，表示对应数字的概率，
    对于真实的标签对应的位置，为1，其他位置为0。
    '''
    res = []  # 用于存储转换后的输出结果

    for i in y:
        y_array = np.zeros(10)  # 初始化一行的输出结果为全 0 数组
        y_array[i - 1] = 1      # 将对应数字位置的值设置为 1

        res.append(y_array)     # 将一行的输出结果添加到 res 中

    return np.array(res)        # 将结果转换为 numpy 数组并返回

    # from sklearn.preprocessing import OneHotEncoder
    # encoder = OneHotEncoder(sparse=False)
    # y_onehot = encoder.fit_transform(y)
    # y_onehot.shape #这个函数与expand_y(y)一致

def expand_array(arr):
    '''
    将一个一维数组复制成一个矩阵，行数等于数组的长度。
    [1, 2, 3]

    [[1, 2, 3],
     [1, 2, 3],
     [1, 2, 3]]
    '''
    # 全为1，长度为数组长度的列向量 @ 原始数组构成的矩阵
    return np.array(np.matrix(np.ones(arr.shape[0])).T @ np.matrix(arr))

def serialize(a, b):
    '''
    将神经网络参数进行序列化（编码）
    '''
    return np.concatenate((np.ravel(a), np.ravel(b)))  # 将两个参数a和b沿着水平方向拼接成一个一维向量

def deserialize(seq):
    '''
    将神经网络参数进行反序列化（解码）
    '''
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)  # 将一个一维向量解码成两个矩阵

def sigmoid(z):
    '''
    逻辑函数
    '''
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    '''
    逻辑函数的导数
    '''
    return np.multiply(sigmoid(z), 1 - sigmoid(z))

def feed_forward(theta, X):
    '''
    对神经网络进行前向传播计算
    '''
    t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]  # 样本数

    # 输入层
    a1 = X          # 5000 * 401

    # 隐藏层
    z2 = a1 @ t1.T  # 5000 * 25
    a2 = np.insert(sigmoid(z2), 0, np.ones(m), axis=1)  # 5000 * 26

    # 输出层
    z3 = a2 @ t2.T   # 5000 * 10
    h = sigmoid(z3)  # 5000 * 10

    return a1, z2, a2, z3, h

def cost(theta, X, y):
    '''
    计算神经网络的代价函数
    '''
    m = X.shape[0]  # 样本数

    _, _, _, _, h = feed_forward(theta, X)  # 前向传播

    # 计算代价函数
    # np.multiply 是逐个元素相乘的操作
    pair_computation = -np.multiply(y, np.log(h + 1e-10)) - np.multiply((1 - y), np.log(1 - h + 1e-10))
    return pair_computation.sum() / m

def regularized_cost(theta, X, y, lambd=1):
    '''
    计算神经网络的正则化代价函数
    '''
    t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]  # 样本数

    reg_t1 = (lambd / (2 * m)) * np.power(t1[:, 1:], 2).sum()  # 计算正则项
    reg_t2 = (lambd / (2 * m)) * np.power(t2[:, 1:], 2).sum()  # 计算正则项

    return cost(theta, X, y) + reg_t1 + reg_t2

def gradient(theta, X, y):
    '''
    计算神经网络的反向传播梯度
    '''
    # 初始化
    t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]               # 样本数
    delta1 = np.zeros(t1.shape)  # (25, 401)
    delta2 = np.zeros(t2.shape)  # (10, 26)

    # 前向传播
    a1, z2, a2, z3, h = feed_forward(theta, X)

    for i in range(m):
        a1i = a1[i, :]  # (1, 401)
        z2i = z2[i, :]  # (1, 25)
        a2i = a2[i, :]  # (1, 26)

        hi = h[i, :]    # (1, 10)
        yi = y[i, :]    # (1, 10)

        # 第3层 输出层
        d3i = hi - yi  # (1, 10)

        # 第2层 隐藏层
        z2i = np.insert(z2i, 0, np.ones(1))  # make it (1, 26) to compute d2i
        d2i = np.multiply(t2.T @ d3i, sigmoid_gradient(z2i))  # (1, 26)

        # 反向传播
        delta2 += np.matrix(d3i).T @ np.matrix(a2i)      # (1, 10).T @ (1, 26) -> (10, 26)
        delta1 += np.matrix(d2i[1:]).T @ np.matrix(a1i)  # (1, 25).T @ (1, 401) -> (25, 401)

    delta1 = delta1 / m
    delta2 = delta2 / m

    return serialize(delta1, delta2)

def regularized_gradient(theta, X, y, lambd=1):
    '''
    计算神经网络的反向传播正则化梯度
    '''
    m = X.shape[0]  # 样本数
    delta1, delta2 = deserialize(gradient(theta, X, y))  # 不带正则项的梯度
    t1, t2 = deserialize(theta)  # 解码为矩阵

    t1[:, 0] = 0                    # 对偏置项的theta进行0处理，不参与正则化
    reg_term_d1 = (lambd / m) * t1  # 计算正则化项对第一层的影响
    delta1 = delta1 + reg_term_d1   # 更新delta1

    t2[:, 0] = 0                    # 对偏置项的theta进行0处理，不参与正则化
    reg_term_d2 = (lambd / m) * t2  # 计算正则化项对第二层的影响
    delta2 = delta2 + reg_term_d2   # 更新delta1

    return serialize(delta1, delta2)  # 将delta1和delta2序列化并返回

def random_init(size):
    '''
    随机初始化参数
    '''
    # uniform函数用于从一个均匀分布中随机采样生成数据
    # 使用均匀分布可以帮助保持初始权重的随机性，从而避免模型收敛到局部极小值
    return np.random.uniform(-0.12, 0.12, size)

def nn_training(X, y):
    '''
    训练神经网络
    '''
    # 初始化参数
    init_theta = random_init(10285)  # 25*401 + 10*26

    # 使用优化器求解参数
    res = opt.minimize(fun=regularized_cost,      # 目标函数
                       x0=init_theta,             # 初始参数
                       args=(X, y, 1),            # 目标函数需要的其他参数
                       method='TNC',              # 优化方法
                       jac=regularized_gradient,  # 目标函数的梯度
                       options={'maxfun': 400})   # 额外参数
    return res

def show_accuracy(theta, X, y):
    '''
    计算神经网络的分类准确率
    '''
    # 调用前向传播函数，获取预测结果
    _, _, _, _, h = feed_forward(theta, X)
    # 根据输出h，对每个样本进行预测，返回每个样本的预测结果
    y_pred = np.argmax(h, axis=1) + 1

    # 输出分类准确率的详细信息
    print(classification_report(y, y_pred))
    
    # 计算总体的准确率，并输出
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print ('总体正确率为：{0}%'.format(accuracy * 100))


'''查看数据集'''

# X, _ = load_data('data2.mat')
# plot_100_image(X)
# plt.show()


'''加载数据'''

X_raw, y_raw = load_data('data2.mat', transpose=False)
X = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)
y = expand_y(y_raw)

# print(X.shape)
# print(y.shape)


'''训练模型'''

res = nn_training(X, y)  # 运行耗时，耐心等待


'''显示准确率'''

final_theta = res.x
_, y_answer = load_data('data2.mat')
show_accuracy(final_theta, X, y_answer)


'''显示隐藏层'''

plot_hidden_layer(final_theta)
plt.show()
