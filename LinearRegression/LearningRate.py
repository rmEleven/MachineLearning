from functions import *
import numpy as np
import matplotlib.pyplot as plt


'''生成一组候选的学习率列表'''
base = np.logspace(-1, -5, num=4)
candidate = np.sort(np.concatenate((base, base*3)))
# print(base)
# print(candidate)


'''读取数据文件并为列命名'''
df = pd.read_csv('data2.txt', names=['square', 'bedrooms', 'price'])
data = normalize_feature(df)


'''训练过程'''
X = get_X(data)
y = get_y(data)
theta = np.zeros(X.shape[1])
epoch = 100

fig, ax = plt.subplots(figsize=(6,6))

for alpha in candidate:
    _, cost_data = batch_gradient_descent(theta, X, y, epoch, alpha=alpha)
    ax.plot(np.arange(len(cost_data)), cost_data, label=alpha)


'''代价训练可视化'''
ax.set_xlabel('epoch', fontsize=12)
ax.set_ylabel('cost', fontsize=12)
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
ax.set_title('learning rate', fontsize=12)
plt.show()