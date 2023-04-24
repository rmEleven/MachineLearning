from functions import *
import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


'''设置seaborn的上下文、样式和调色板'''
sns.set(context="notebook", style="whitegrid", palette="dark")


'''读取数据文件并为列命名'''
df = pd.read_csv('data2.txt', names=['square', 'bedrooms', 'price'])
data = normalize_feature(df)
# print(raw_data.head())
# raw_data.info()
# raw_data.describe()
# print(data.head())


'''训练过程'''
X = get_X(data)
y = get_y(data)
theta = np.zeros(X.shape[1])
alpha = 0.01
epoch = 500
final_theta, cost_data = batch_gradient_descent(theta, X, y, epoch, alpha=alpha)
theta0 = final_theta[0]
theta1 = final_theta[1]
theta2 = final_theta[2]

'''数据可视化'''
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,1], X[:,2], y, label='Training Data')
ax.legend(loc=2)

ax.set_xlabel('square')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
ax.set_title('square & bedrooms vs. price')

plt.show()


'''代价数据可视化'''
sns.lineplot(x=np.arange(len(cost_data)), y=cost_data)
plt.xlabel('epoch', fontsize=18)
plt.ylabel('cost', fontsize=18)
plt.show()


'''my model预测表现'''
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:,1], X[:,2], theta0 + theta1*data.square + theta2*data.bedrooms, 'r', label='Prediction')
ax.scatter(X[:,1], X[:,2], y, label='Training Data')
ax.legend(loc=2)

ax.set_xlabel('square')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
ax.set_title('square & bedrooms vs. price: my model')

plt.show()


'''scikit-learn model预测表现'''
model = linear_model.LinearRegression()
model.fit(X, y)
f = model.predict(X).flatten()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:,1], X[:,2], f, 'r', label='Prediction')
ax.scatter(X[:,1], X[:,2], y, label='Training Data')
ax.legend(loc=2)

ax.set_xlabel('square')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
ax.set_title('square & bedrooms vs. price: scikit-learn model')

plt.show()