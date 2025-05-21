import numpy as np
from sklearn.datasets import load_iris

# 加载前100条（两类）
iris = load_iris()
X = iris.data[:100]
y = iris.target[:100]

# 添加偏置项
X = np.hstack([np.ones((X.shape[0],1)), X]) # 加一列偏置项b=1


# sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 训练逻辑回归
def train_logistic_regression(X, y, lr=0.1, epochs=1000):
    weights = np.zeros(X.shape[1])  # 初始化权重
    for _ in range(epochs):
        z = np.dot(X, weights)  # 计算z=wx
        predictions = sigmoid(z)
        # 计算梯度
        gradient = np.dot(X.T, (predictions - y)) / len(y)
        weights -= lr * gradient
    return weights


# 预测函数
def predict(X, weights):
    probs = sigmoid(np.dot(X, weights))
    return (probs >= 0.5).astype(int)


weights = train_logistic_regression(X, y)

# 测试整体准确率
y_pred = predict(X, weights)
accuracy = np.mean(y_pred == y)
print(f"逻辑回归训练准确率: {accuracy:.2f}")


# 测试单个样本
for i in range(5):
    pred = predict(X[i:i+1], weights)
    print(f"样本{i}预测标签: {pred[0]}, 真实标签: {y[i]}")

for i in range(96,100):
    pred = predict(X[i:i+1], weights)
    print(f"样本{i}预测标签: {pred[0]}, 真实标签: {y[i]}")