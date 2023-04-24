from functions import *
import matplotlib.pyplot as plt


'''读取数据文件并为列命名'''
df = pd.read_csv('data2.txt', names=['square', 'bedrooms', 'price'])
data = normalize_feature(df)


'''训练过程'''
X = get_X(data)
y = get_y(data)

final_theta = normalEqn(X, y)
f = final_theta[0] + final_theta[1] * X[:, 1] + final_theta[2] * X[:, 2]


'''训练结果可视化'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 1], X[:, 2], f, 'r', label='Prediction')
ax.scatter(X[:, 1], X[:, 2], y, label='Traning Data')
ax.legend(loc=2)

ax.set_xlabel('square')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
ax.set_title('square & bedrooms vs. price')

plt.show()
