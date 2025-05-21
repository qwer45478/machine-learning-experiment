from sklearn.datasets import load_iris
import math
import numpy as np
from collections import Counter

# 加载 iris 数据集（仅取前 100 条用于二分类）
iris = load_iris()
X_raw = iris.data[:100]
y_raw = iris.target[:100]  # 只取 0 和 1 类（Setosa 和 Versicolor）


# 离散化处理（为简化 ID3，这里将连续值分成高/低两类）
def discretize_features(X):
    x_disc = []
    for col in X.T:
        mean_val = np.mean(col)
        x_disc.append(['high' if val > mean_val else 'low' for val in col])
    return list(map(list, zip(*x_disc)))  # 转置回来


X = discretize_features(X_raw)
y = ['versicolor' if label == 1 else 'sentosa' for label in y_raw]

# 将数据组合成 dataset
dataset = [x + [label] for x, label in zip(X, y)]
feature_names = iris.feature_names[:4]


# 计算熵不纯度
def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


# 计算信息增益=总熵-加权熵
def info_gain(dataset, feature_index):
    total_entropy = entropy([row[-1] for row in dataset])
    values = set(row[feature_index] for row in dataset)
    weighted_entropy = 0
    for val in values:
        subset = [row for row in dataset if row[feature_index] == val]
        subset_labels = [row[-1] for row in subset]
        weighted_entropy += (len(subset) / len(dataset)) * entropy(subset_labels)
    return total_entropy - weighted_entropy


def majority_class(labels):
    return Counter(labels).most_common(1)[0][0]


def build_tree(dataset, feature_names):
    labels = [row[-1] for row in dataset]
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if not feature_names:
        return majority_class(labels)

    # 选择信息增益最大的特征
    gains = [info_gain(dataset, i) for i in range(len(feature_names))]
    best_index = gains.index(max(gains))
    best_feature = feature_names[best_index]

    tree = {best_feature: {}}
    values = set(row[best_index] for row in dataset)

    for val in values:
        subset = [row[:best_index] + row[best_index+1:] for row in dataset if row[best_index] == val]
        sub_labels = feature_names[:best_index] + feature_names[best_index+1:]
        subtree = build_tree([r for r in subset], sub_labels)
        tree[best_feature][val] = subtree

    return tree


def predict(tree, sample, feature_names):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    index = feature_names.index(feature)
    feature_value = sample[index]
    subtree = tree[feature].get(feature_value)
    return predict(subtree, sample, feature_names)


tree = build_tree(dataset, feature_names)
print("构建的决策树：")
print(tree)

# 用前 5 个样本测试
for i in range(5):
    pred = predict(tree, X[i], feature_names)
    print(f"样本{i} 实际: {y[i]}，预测: {pred}")

for i in range(96, 100):
    pred = predict(tree, X[i], feature_names)
    print(f"样本{i} 实际: {y[i]}，预测: {pred}")
