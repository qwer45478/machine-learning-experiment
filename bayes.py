import numpy as np
from sklearn.datasets import load_iris
import math
from collections import defaultdict

# 加载 Iris 数据集
iris = load_iris()
X_all = iris.data[:100]
y_all = iris.target[:100]

# 划分训练集与测试集（前10个0类，后10个1类做测试）
X_test = np.vstack((X_all[:10], X_all[-10:]))     # 10个Setosa + 10个Versicolor
y_test = np.concatenate((y_all[:10], y_all[-10:]))

X_train = X_all[10:-10]
y_train = y_all[10:-10]


# 计算均值
def mean(numbers):
    return sum(numbers) / len(numbers)


# 计算方差
def var(numbers):
    avg = mean(numbers)
    return sum((x - avg) ** 2 for x in numbers) / len(numbers)


# 高斯概率密度函数
def gaussian_prob(x, mean, var):
    if var == 0:
        return 1.0 if x == mean else 0.0
    exponent = math.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / (math.sqrt(2 * math.pi) * var)) * exponent


# 训练朴素贝叶斯模型
def train_naive_bayes(X, y):
    separated = defaultdict(list)
    for xi, yi in zip(X, y):    # zip()将二维数组按列打包，即对每个特征维度分别取出所有值，计算其均值和标准差
        separated[yi].append(xi)
    model = {}
    for cls, samples in separated.items():
        summaries = [(mean(feature), var(feature)) for feature in zip(*samples)]
        model[cls] = summaries
    return model


# 预测函数
def predict_naive_bayes(model, x):
    probs = {}
    for cls, summaries in model.items():
        probs[cls] = 1
        for i in range(len(x)):
            mean_, var_ = summaries[i]
            probs[cls] *= gaussian_prob(x[i], mean_, var_)
    return max(probs, key=probs.get)


# 模型训练
nb_model = train_naive_bayes(X_train, y_train)

# 模型预测
print("朴素贝叶斯预测结果：")
for xi, yi in zip(X_test, y_test):
    pred = predict_naive_bayes(nb_model, xi)
    print(f"真实值: {yi}, 预测值: {pred}")

