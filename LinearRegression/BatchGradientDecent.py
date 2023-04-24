from functions import *
import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt


'''设置seaborn的上下文、样式和调色板'''
sns.set(context="notebook", style="whitegrid", palette="dark")


'''读取数据文件并为列命名'''
df = pd.read_csv('data1.txt', names=['population', 'profit'])
data = df
# print(df.head())  # 显示数据前五行
# df.info()         # 打印df的class信息
# df.describe()     # 打印df的统计信息


'''训练过程'''
X = get_X(data)
y = get_y(data)
theta = np.zeros(X.shape[1])
alpha = 0.01
epoch = 500
final_theta, cost_data = batch_gradient_descent(theta, X, y, epoch, alpha=alpha)
b = final_theta[0]
m = final_theta[1]


'''数据可视化'''
sns.lmplot(x='population', y='profit', data=df, height=6, fit_reg=False)
plt.show()


'''代价数据可视化'''
ax = sns.lineplot(x=np.arange(len(cost_data)), y=cost_data)
ax.set_xlabel('epoch')
ax.set_ylabel('cost')
plt.show()


'''my model预测表现'''
plt.scatter(data.population, data.profit, label="Training data")
plt.plot(data.population, data.population*m + b, 'r', label="Prediction")
plt.legend(loc=2)
plt.show()


'''scikit-learn model预测表现'''
model = linear_model.LinearRegression()
model.fit(X, y)
f = model.predict(X).flatten()
plt.scatter(data.population, data.profit, label='Training Data')
plt.plot(data.population, f, 'r', label='Prediction')
plt.legend(loc=2)
plt.show()
