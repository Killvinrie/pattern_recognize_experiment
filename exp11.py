import numpy as np

# 定义计算两点之间的欧氏距离的函数
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# K-means 聚类算法
def kmeans(X, k, max_iters=100):
    # 随机初始化质心（从数据点中随机选择 k 个点）
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for i in range(max_iters):
        # 创建一个空的列表来保存每个数据点的簇标签
        labels = np.zeros(X.shape[0])

        # 对每个数据点找到最近的质心
        for idx, point in enumerate(X):
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            labels[idx] = np.argmin(distances)

        # 保存旧的质心以检查是否收敛
        old_centroids = centroids.copy()

        # 计算新的质心，基于每个簇的平均值
        for j in range(k):
            points_in_cluster = X[labels == j]
            if len(points_in_cluster) > 0:
                centroids[j] = np.mean(points_in_cluster, axis=0)

        # 检查质心是否发生变化，如果没有变化则说明算法收敛
        if np.all(centroids == old_centroids):
            break

    return labels, centroids

# 测试数据集
data = np.array([
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0]
])

# 聚类数量 k
k = 2

# 运行 K-means 算法
labels, centroids = kmeans(data, k)

# 打印结果
print("聚类标签：", labels)
print("聚类中心：", centroids)