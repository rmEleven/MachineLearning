from functions import *
import numpy as np
import matplotlib.pyplot as plt

base = np.logspace(-1, -5, num=4)
candidate = np.sort(np.concatenate((base, base*3)))
# print(base)
# print(candidate)

'''读取数据文件并为列命名'''
raw_data = pd.read_csv('data2.txt', names=['square', 'bedrooms', 'price'])
data = normalize_feature(raw_data)

X = get_X(data)
y = get_y(data)
theta = np.zeros(X.shape[1])
epoch = 50

fig, ax = plt.subplots(figsize=(8,8))

for alpha in candidate:
    _, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha=alpha)
    ax.plot(np.arange(len(cost_data)), cost_data, label=alpha)

ax.set_xlabel('epoch', fontsize=12)
ax.set_ylabel('cost', fontsize=12)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('learning rate', fontsize=12)
plt.show()