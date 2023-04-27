from sklearn.metrics import classification_report
from sklearn import linear_model

from functions import *


'''读取数据'''

# 读取一个名为 data2.txt 的文本文件
# 读取的数据被存储在名为 data 的 Pandas DataFrame 对象中
# 为读取的 DataFrame 指定列名
data = pd.read_csv('data2.txt', names=['test1', 'test2', 'accepted'])

print(data.head())      # 显示前5条数据
print(data.describe())  # 显示统计信息


'''数据可视化'''

positive = data[data['accepted'].isin([1])]  # 筛选正向类
negative = data[data['accepted'].isin([0])]  # 筛选负向类

fig, ax = plt.subplots(figsize=(12,8))  # 创建一个图形对象和坐标轴对象，并设置图形大小为12x8英寸。

# 绘制正样本的散点图，x轴为test1特征，y轴为test2特征，散点大小为50，颜色为蓝色，形状为圆圈，并添加图例标签Accepted。
ax.scatter(positive['test1'], positive['test2'], s=50, c='b', marker='o', label='Accepted')

# 绘制负样本的散点图，x轴为exam1特征，y轴为exam2特征，散点大小为50，颜色为红色，形状为叉，并添加图例标签Not Admitted。
ax.scatter(negative['test1'], negative['test2'], s=50, c='r', marker='x', label='Rejected')

ax.legend()  # 添加图例
ax.set_xlabel('Test 1 Score')  # 设置x轴标签为Exam1 Score
ax.set_ylabel('Test 2 Score')  # 设置y轴标签为Exam2 Score
ax.set_title('Data Visualization')  # 设置图形标题为Data Visualization
plt.show()  # 显示图形


'''拟合参数'''

x1 = np.array(data.test1)
x2 = np.array(data.test2)
X = feature_mapping(x1, x2, power=6, as_ndarray=True)  # 获取多项式特征
y = get_y(data)
theta = np.zeros(X.shape[1])
learningRate = 1

# 使用scipy库中的optimize模块中的minimize函数来最小化代价函数cost，得到适合的参数值theta
# fun：要最小化的目标函数（即代价函数）
# x0：代价函数的参数值的初始估计
# args：代价函数中需要传入的其他参数
# method：使用的优化算法的名称
# jac：代价函数的梯度（导数）
res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y), method='Newton-CG', jac=regularized_gradient)
# message: 终止状态的描述信息
# success: 优化是否成功
# status: 优化的终止状态
# fun: 最优解对应的目标函数值
# x: 最优解向量
# nit: 迭代次数
# jac: 最优解对应的梯度向量
# nfev: 目标函数调用次数
# njev: 梯度调用次数
# nhev: Hessian调用次数
# hess_inv: 最优解对应的Hessian矩阵的逆的线性算子
final_theta = res.x


'''模型验证'''

predictions = predict(final_theta, X)

# precision：精确度，指分类器预测为正类的样本中实际为正类的比例。
# recall：召回率，指实际为正类的样本中被分类器正确预测为正类的比例。
# f1-score：F1分数是精度和召回率的加权平均值，用于综合评估分类器的性能。
# support：支持度，指在测试集中属于这个类别的样本数量。
print(classification_report(y, predictions))  # 打印分类模型的性能报告
# accuracy：准确度，指分类器正确分类的样本数与总样本数之比。
# macro avg：宏平均值，将指标在所有类别上取平均值，不考虑类别不平衡问题。
# weighted avg：加权平均值，将指标在所有类别上取加权平均值，考虑类别不平衡问题。


'''调用线性回归包'''

# 创建逻辑回归模型对象 表示使用L2正则化 正则化系数为1.0
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
# 训练逻辑回归模型 X是训练集的特征矩阵 y是训练集的标签 .ravel()将y转换为一维数组
model.fit(X, y.ravel())
# 打印模型在训练集上的准确率
print(model.score(X, y))

draw_boundary(data, power=6, l=1)    # lambda=1
draw_boundary(data, power=6, l=0)    # lambda=0，过拟合
draw_boundary(data, power=6, l=100)  # lambda=100，欠拟合
