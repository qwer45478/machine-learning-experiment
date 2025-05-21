import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 1) 数据中心化
X_centered = X - np.mean(X, axis=0)

# 2) 协方差矩阵
cov_matrix = np.cov(X_centered, rowvar=False)

# 3) 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 4) 特征值排序（降序）
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# 5) 选取前 k 个主成分
k = 2
components = eigenvectors_sorted[:, :k]

# 6) 降维
X_reduced = np.dot(X_centered, components)

# 可视化
colors = ['red', 'green', 'blue']
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
    plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1],
                color=colors[i], label=target_name, alpha=0.7)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of IRIS Dataset')
plt.legend()
plt.grid(True)
plt.show()
