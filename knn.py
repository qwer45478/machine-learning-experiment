from sklearn.datasets import load_iris
import math
from collections import Counter
import numpy as np

iris = load_iris()
X_all = iris.data[:100]       # 只用类别 0 和 1（Setosa 和 Versicolor）
y_all = iris.target[:100]     # 标签：0 或 1

# 划分训练集与测试集（前10个0类，后10个1类做测试）
X_test = np.vstack((X_all[:10], X_all[-10:]))     # 10个Setosa + 10个Versicolor
y_test = np.concatenate((y_all[:10], y_all[-10:]))

X_train = X_all[10:-10]
y_train = y_all[10:-10]


def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def predict_knn(X_train, y_train, x_test, k):
    distances = [(euclidean_distance(x_test, xi), yi) for xi, yi in zip(X_train, y_train)]
    distances.sort(key=lambda tup: tup[0])  # 按距离排序,使用匿名函数指定返回的参数为tup[0]
    k_labels = [label for _, label in distances[:k]]  # 取前k个标签
    return Counter(k_labels).most_common(1)[0][0]  # 返回出现最多的标签



print("K近邻预测结果：")
for xi, yi in zip(X_test, y_test):
    pred = predict_knn(X_train, y_train, xi, 3)     # k = 3
    print(f"真实: {yi}, 预测: {pred}")
