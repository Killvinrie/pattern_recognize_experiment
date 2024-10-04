import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# 1. 加载数据集
def load_data(file_path):
    """加载并返回数据集"""
    data = pd.read_csv(file_path)
    return data.values  # 将数据转换为 numpy 数组

# 2. 实现 K-Means 聚类算法
class KMeans:
    def __init__(self, k=10, max_iters=100):
        """初始化 KMeans 参数"""
        self.k = k  # 聚类的个数
        self.max_iters = max_iters  # 最大迭代次数
    
    def initialize_centroids(self, X):
        """随机初始化质心"""
        random_indices = np.random.choice(len(X), self.k, replace=False)
        return X[random_indices]
    
    def assign_clusters(self, X, centroids):
        """将每个点分配到最近的质心"""
        distances = cdist(X, centroids, 'euclidean')  # 计算欧氏距离
        return np.argmin(distances, axis=1)  # 返回最小距离的质心索引
    
    def update_centroids(self, X, labels):
        """根据分配的簇更新质心"""
        new_centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            points_in_cluster = X[labels == i]
            if len(points_in_cluster) > 0:
                new_centroids[i] = points_in_cluster.mean(axis=0)
        return new_centroids
    
    def fit(self, X):
        """执行 K-Means 聚类算法"""
        centroids = self.initialize_centroids(X)
        for _ in range(self.max_iters):
            labels = self.assign_clusters(X, centroids)
            new_centroids = self.update_centroids(X, labels)
            if np.all(centroids == new_centroids):
                break  # 质心不再变化时停止
            centroids = new_centroids
        self.centroids = centroids
        self.labels = labels
        return self
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# 1. 展示标签为 1 的 200 幅图像
def display_images_by_label(X, labels, target_label=1, num_images=200):
    """根据标签展示图像"""
    selected_images = X[labels == target_label]  # 筛选出标签为 target_label 的图像
    selected_images = selected_images[:num_images]  # 选择前 num_images 个图像
    
    # 设置网格大小 (20行 x 10列)
    rows, cols = 20, 10
    fig, axes = plt.subplots(rows, cols, figsize=(15, 30))  # 设定图片大小
    
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            ax.imshow(selected_images[i * cols + j].reshape(28, 28), cmap='gray')
            ax.axis('off')  # 关闭坐标轴显示
    plt.tight_layout()  # 调整布局，避免图像重叠
    plt.show()

# 2. 运行 K-Means 聚类并展示标签为 1 的图像
if __name__ == "__main__":
    # 指定文件路径并加载数据
    file_path = 'Sample.csv'  # 替换为你实际的文件路径
    X = load_data(file_path)
    
    # 初始化 KMeans 模型并进行聚类
    kmeans = KMeans(k=10, max_iters=100)
    kmeans.fit(X)
    
    # 打印前 10 个样本的聚类标签
    print("Cluster assignments for first 10 samples:", kmeans.labels[:10])
    
    # 展示标签为 1 的 200 幅图像
    display_images_by_label(X, kmeans.labels, target_label=1, num_images=200)
