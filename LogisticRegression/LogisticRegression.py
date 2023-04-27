from sklearn.metrics import classification_report

from functions import *


'''读取数据'''

# 读取一个名为 data1.txt 的文本文件
# 读取的数据被存储在名为 data 的 Pandas DataFrame 对象中
# 指定数据文件没有列名，将第一行数据视为数据而非列名
# 为读取的 DataFrame 指定列名
data = pd.read_csv('data1.txt', header=None, names=['exam1', 'exam2', 'admitted'])

print(data.head())      # 显示前5条数据
print(data.describe())  # 显示统计信息


'''数据可视化'''

positive = data[data['admitted'].isin([1])]  # 筛选正向类
negative = data[data['admitted'].isin([0])]  # 筛选负向类

fig, ax = plt.subplots(figsize=(12, 8))  # 创建一个图形对象和坐标轴对象，并设置图形大小为12x8英寸。

# 绘制正样本的散点图，x轴为exam1特征，y轴为exam2特征，散点大小为50，颜色为蓝色，形状为圆圈，并添加图例标签Admitted。
ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', marker='o', label='Admitted')

# 绘制负样本的散点图，x轴为exam1特征，y轴为exam2特征，散点大小为50，颜色为红色，形状为叉，并添加图例标签Not Admitted。
ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='x', label='Not Admitted')

ax.legend()  # 添加图例
ax.set_xlabel('Exam1 Score')  # 设置x轴标签为Exam1 Score
ax.set_ylabel('Exam2 Score')  # 设置y轴标签为Exam2 Score
ax.set_title('Data Visualization')  # 设置图形标题为Data Visualization
plt.show()  # 显示图形


'''测试sigmoid函数'''

nums = np.arange(-10, 10, step=1)  # 生成一个从-10到10的等差数列，步长为1
fig, ax = plt.subplots(figsize=(12,8))  # 创建一个12x8的图像，并返回包含图像和轴的元组
ax.plot(nums, sigmoid(nums), 'orange')  # 绘制sigmoid函数曲线，nums为横坐标，sigmoid(nums)为纵坐标，颜色为橙色。
ax.set_xlabel('z', fontsize=18)  # 设置x轴标签为'z'，字体大小为18。
ax.set_ylabel('g(z)', fontsize=18)  # 设置y轴标签为'g(z)'，字体大小为18。
ax.set_title('sigmoid function', fontsize=18)  # 设置图像标题为'sigmoid function'，字体大小为18。
plt.show()  # 显示图形


'''拟合参数'''

X = get_X(data)
y = get_y(data)
theta = np.zeros(3)

# 使用scipy库中的optimize模块中的minimize函数来最小化代价函数cost，得到适合的参数值theta
# fun：要最小化的目标函数（即代价函数）
# x0：代价函数的参数值的初始估计
# args：代价函数中需要传入的其他参数
# method：使用的优化算法的名称
# jac：代价函数的梯度（导数）
res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)
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

predictions = predict(final_theta, X)  # 预测结果的向量
correct = [1 if(a == b) else 0 for (a, b) in zip(predictions, y)]  # 模型的预测是否正确
accuracy = (sum(map(int, correct))) % len(correct)  # 将正确率转换为百分比格式
print('accuracy = {0}%'.format(accuracy))  # 打印正确率

# precision：精确度，指分类器预测为正类的样本中实际为正类的比例。
# recall：召回率，指实际为正类的样本中被分类器正确预测为正类的比例。
# f1-score：F1分数是精度和召回率的加权平均值，用于综合评估分类器的性能。
# support：支持度，指在测试集中属于这个类别的样本数量。
print(classification_report(y, predictions))  # 打印分类模型的性能报告
# accuracy：准确度，指分类器正确分类的样本数与总样本数之比。
# macro avg：宏平均值，将指标在所有类别上取平均值，不考虑类别不平衡问题。
# weighted avg：加权平均值，将指标在所有类别上取加权平均值，考虑类别不平衡问题。


'''寻找决策边界'''

# theta0 + theta1 * x1 + theta2 * x2 = 0
x1 = np.arange(130, step=0.1)
x2 = -final_theta[0] / final_theta[2] - final_theta[1] / final_theta[2] * x1

# 创建一个带有决策边界的散点图
# 横轴的变量名 纵轴的变量名 颜色变量 数据集 图形的高度 图片的宽高比 是否绘制线性回归线 散点的属性
sns.lmplot(x='exam1', y='exam2', hue='admitted', data=data, 
           height=6, aspect=1.5, fit_reg=False, scatter_kws={'s': 25})

plt.plot(x1, x2, 'grey')  # 绘制灰色的线性决策边界
plt.xlim(0, 130)  # 设置x轴的范围
plt.ylim(0, 130)  # 设置y轴的范围
plt.title('Decision Boundary')  # 设置图表的标题
plt.show()  # 显示图表
