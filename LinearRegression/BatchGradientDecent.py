from functions import *
import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt

'''设置seaborn的上下文、样式和调色板'''
sns.set(context="notebook", style="whitegrid", palette="dark")

'''读取数据文件并为列命名'''
df = pd.read_csv('data1.txt', names=['population', 'profit'])
# print(df.head())  # 显示数据前五行
# df.info()         # 打印df的class信息
# df.describe()     # 打印df的统计信息

'''训练过程'''
data = df
X = get_X(data)
y = get_y(data)
theta = np.zeros(X.shape[1])
epoch = 500
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch)

'''数据可视化'''
sns.lmplot(x='population', y='profit', data=df, height=6, fit_reg=False)
plt.show()

'''代价数据可视化'''
ax = sns.lineplot(x=np.arange(len(cost_data)), y=cost_data)
ax.set_xlabel('epoch')
ax.set_ylabel('cost')
plt.show()

'''my model预测表现'''
b = final_theta[0]
m = final_theta[1]
plt.scatter(data.population, data.profit, label="Training data")
plt.plot(data.population, data.population*m+b, 'r', label="Prediction")
plt.legend(loc=2)
plt.show()

'''scikit-learn model预测表现'''
model = linear_model.LinearRegression()
model.fit(X, y)

x = X[:, 1]
f = model.predict(X).flatten()

plt.scatter(X[:, 1], y, label='Training Data')
plt.plot(x, f, 'r', label='Prediction')
plt.legend(loc=2)
plt.show()
