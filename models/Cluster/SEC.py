import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons, make_circles

def build_sparse_graph(data, k=5, threshold=0.5):
    """
    构建稀疏图
    :param data: 数据集
    :param k: k-NN的k值
    :param threshold: 距离阈值
    :return: 稀疏图
    """
    if k > len(data):
        k = len(data) - 1  # 确保k值不超过样本数
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    rows = np.repeat(np.arange(len(data)), k)
    cols = indices.flatten()
    weights = np.exp(-distances.flatten() ** 2 / (2 * threshold ** 2))
    sparse_graph = csr_matrix((weights, (rows, cols)), shape=(len(data), len(data)))
    return sparse_graph

def calculate_structural_entropy(sparse_graph):
    """
    计算结构熵
    :param sparse_graph: 稀疏图
    :return: 结构熵值
    """
    degree_matrix = np.array(sparse_graph.sum(axis=1)).flatten()
    prob = degree_matrix / degree_matrix.sum()
    return entropy(prob)

def build_coding_tree(sparse_graph):
    """
    构建编码树
    :param sparse_graph: 稀疏图
    :return: 编码树
    """
    n_components, labels = connected_components(csgraph=sparse_graph, directed=False, return_labels=True)
    return labels

def iterative_pre_delete_and_reassign(data, labels, threshold=0.5):
    """
    迭代预删除和重新分配
    :param data: 数据集
    :param labels: 初始标签
    :param threshold: 距离阈值
    :return: 优化后的标签
    """
    n_samples = len(data)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if np.linalg.norm(data[i] - data[j]) < threshold:
                if labels[i] != labels[j]:
                    labels[j] = labels[i]
    return labels

def sec_algorithm(data, k=5, threshold=0.5):
    """
    SEC算法主函数
    :param data: 数据集
    :param k: k-NN的k值
    :param threshold: 距离阈值
    :return: 聚类结果
    """
    sparse_graph = build_sparse_graph(data, k, threshold)
    structural_entropy_value = calculate_structural_entropy(sparse_graph)
    coding_tree_labels = build_coding_tree(sparse_graph)
    optimized_labels = iterative_pre_delete_and_reassign(data, coding_tree_labels, threshold)
    return optimized_labels

