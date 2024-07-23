import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from sklearn.impute import SimpleImputer

# 计算邻域一阶连接矩阵
def calculate_first_order_connectivity(ca_matrix, kernel_function):
    """
    计算邻域一阶连接矩阵。

    param
    ca_matrix (torch.Tensor): 邻域矩阵。
    kernel_function (function): 核函数，用于计算连接权重。

    return: 
    torch.Tensor: 一阶连接矩阵。
    """
    n = ca_matrix.shape[0]
    first_order_connectivity = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if ca_matrix[i, j] != 0:
                first_order_connectivity[i, j] = kernel_function(i, j)
    return first_order_connectivity

# 构建高阶连接矩阵
def build_high_order_connection_matrix(first_order_connectivity, order):
    """
    构建高阶连接矩阵。

    params
    first_order_connectivity (torch.Tensor): 一阶连接矩阵。
    order (int): 高阶的阶数。

    return: 
    torch.Tensor: 高阶连接矩阵。
    """
    high_order_connectivity = first_order_connectivity
    for _ in range(order - 1):
        high_order_connectivity = torch.matmul(high_order_connectivity, first_order_connectivity)
    return high_order_connectivity

# 学习最优连接矩阵
def learn_optimal_connection_matrix(high_order_connection_matrices, alpha):
    """
    学习最优连接矩阵。

    params: 
    high_order_connection_matrices (list of torch.Tensor): 高阶连接矩阵列表。
    alpha (list of float): 权重列表。

    return: 
    torch.Tensor: 最优连接矩阵。
    """
    n = high_order_connection_matrices[0].shape[0]
    C = torch.zeros((n, n))
    for i in range(len(high_order_connection_matrices)):
        C += alpha[i] * high_order_connection_matrices[i]
    return C

# 自适应加权
def adaptive_weighting(C, E, A, alpha, lambda_, high_order_connection_matrices):
    """
    自适应加权。

    params: 
    C (torch.Tensor): 最优连接矩阵。
    E (torch.Tensor): 误差矩阵。
    A (torch.Tensor): 邻域矩阵。
    alpha (list of float): 权重列表。
    lambda_ (float): 正则化参数。
    high_order_connection_matrices (list of torch.Tensor): 高阶连接矩阵列表。

    return: 
    float: 损失值。
    """
    o = len(alpha)
    loss = 0
    for i in range(o):
        loss += torch.norm(C - alpha[i] * high_order_connection_matrices[i], p='fro') ** 2
    loss += lambda_ / 2 * torch.norm(E, p='fro') ** 2
    return loss

# 更新 C
def update_C(alpha, G, Y1, Y2, mu, A, E, J, F, high_order_connection_matrices):
    """
    更新最优连接矩阵 C。

    params: 
    alpha (list of float): 权重列表。
    G (list of torch.Tensor): 高阶连接矩阵列表。
    Y1 (torch.Tensor): 拉格朗日乘子矩阵。
    Y2 (torch.Tensor): 拉格朗日乘子矩阵。
    mu (float): 惩罚参数。
    A (torch.Tensor): 邻域矩阵。
    E (torch.Tensor): 误差矩阵。
    J (torch.Tensor): 辅助矩阵。
    F (torch.Tensor): 辅助矩阵。
    high_order_connection_matrices (list of torch.Tensor): 高阶连接矩阵列表。

    return: 
    torch.Tensor: 更新后的最优连接矩阵 C。
    """
    o = len(alpha)
    C = (2 * o * torch.eye(A.shape[0]) + 2 * mu * torch.eye(A.shape[0])) @ torch.inverse(
        mu * (A - E + Y1 / mu) + mu * (J - Y2 / mu) + F @ F.t() + 2 * torch.sum(torch.stack([alpha[i] * G[i] for i in range(o)]), dim=0))
    return C

# 更新 alpha
def update_alpha(C, G, theta):
    """
    更新权重 alpha。

    params: 
    C (torch.Tensor): 最优连接矩阵。
    G (list of torch.Tensor): 高阶连接矩阵列表。
    theta (float): 参数。

    return: 
    np.ndarray: 更新后的权重 alpha。
    """
    o = len(G)
    alpha = np.zeros(o)
    for i in range(o):
        alpha[i] = 2 * torch.trace(C @ G[i]) - theta / (2 * torch.trace(G[i] @ G[i].t()))
    return alpha

# 更新 J
def update_J(C, Y2, mu):
    """
    更新辅助矩阵 J。

    params: 
    C (torch.Tensor): 最优连接矩阵。
    Y2 (torch.Tensor): 拉格朗日乘子矩阵。
    mu (float): 惩罚参数。

    return: 
    torch.Tensor: 更新后的辅助矩阵 J。
    """
    J = torch.min(torch.max((C + Y2 / mu) / 2 + (C.t() + Y2.t() / mu) / 2, torch.zeros_like(C)), torch.ones_like(C))
    return J

# 更新 E
def update_E(A, C, Y1, lambda_, mu):
    """
    更新误差矩阵 E。

    params: 
    A (torch.Tensor): 邻域矩阵。
    C (torch.Tensor): 最优连接矩阵。
    Y1 (torch.Tensor): 拉格朗日乘子矩阵。
    lambda_ (float): 正则化参数。
    mu (float): 惩罚参数。

    return: 
    torch.Tensor: 更新后的误差矩阵 E。
    """
    E = (mu * (A - C) + Y1) / (lambda_ + mu)
    return E

# 更新 Z
def update_Z(C, Z, H, Lz, gamma):
    """
    更新矩阵 Z。

    params: 
    C (torch.Tensor): 最优连接矩阵。
    Z (torch.Tensor): 矩阵 Z。
    H (torch.Tensor): 辅助矩阵。
    Lz (torch.Tensor): 辅助矩阵。
    gamma (float): 参数。

    return: 
    torch.Tensor: 更新后的矩阵 Z。
    """
    n = C.shape[0]
    D = torch.diag(torch.sum(C, dim=1))
    Z = torch.inverse(C @ D ** (-1 / 2) @ C.t() + gamma * torch.eye(n)) @ C @ D ** (-1 / 2)
    return Z

# ALM 更新 Z
def ALM_update_Z(G_prime, q, b, rho2, eta, max_iterations, epsilon):
    """
    使用 ALM 方法更新矩阵 Z。

    params: 
    G_prime (torch.Tensor): 辅助矩阵。
    q (torch.Tensor): 辅助向量。
    b (torch.Tensor): 辅助向量。
    rho2 (float): 参数。
    eta (float): 参数。
    max_iterations (int): 最大迭代次数。
    epsilon (float): 收敛阈值。

    return: 
    torch.Tensor: 更新后的矩阵 Z。
    """
    n = G_prime.shape[0]
    Z = torch.ones((n, n))

    for iteration in range(max_iterations):
        p = Z - (1 / eta) * (torch.matmul(G_prime.t(), Z) + q)
        Z_prev = Z.clone()
        Z = torch.min(torch.max((p + (1 / eta) * q + (torch.matmul(G_prime, p) - b) / eta).view(n, n), torch.zeros((n, n))), torch.ones((n, n)))
        eta *= rho2

        if torch.norm(Z - Z_prev) < epsilon:
            break

    return Z

# AWEC 算法
def AWEC(CA_matrix, lambda_, gamma, alpha, Y1, Y2, mu, max_iterations, epsilon, high_order_connection_matrices, theta, rho1, mu_max, q, b, rho2, eta):
    """
    AWEC 算法主函数。

    params: 
    CA_matrix (torch.Tensor): 邻域矩阵。
    lambda_ (float): 正则化参数。
    gamma (float): 参数。
    alpha (list of float): 权重列表。
    Y1 (torch.Tensor): 拉格朗日乘子矩阵。
    Y2 (torch.Tensor): 拉格朗日乘子矩阵。
    mu (float): 惩罚参数。
    max_iterations (int): 最大迭代次数。
    epsilon (float): 收敛阈值。
    high_order_connection_matrices (list of torch.Tensor): 高阶连接矩阵列表。
    theta (float): 参数。
    rho1 (float): 参数。
    mu_max (float): 惩罚参数最大值。
    q (torch.Tensor): 辅助向量。
    b (torch.Tensor): 辅助向量。
    rho2 (float): 参数。
    eta (float): 参数。

    return: 
    tuple: 包含最优连接矩阵 C、误差矩阵 E 和矩阵 Z。
    """
    n = CA_matrix.shape[0]
    C = torch.eye(n)
    E = torch.zeros((n, n))
    Z = torch.eye(n)
    J = torch.eye(n)
    D = torch.diag(torch.sum(CA_matrix, dim=1))

    for iteration in range(max_iterations):
        # 更新 C
        F = D ** (-1 / 2) @ Z
        C = update_C(alpha, high_order_connection_matrices, Y1, Y2, mu, CA_matrix, E, J, F, high_order_connection_matrices)

        # 更新 J
        J = update_J(C, Y2, mu)

        # 更新 E
        E = update_E(CA_matrix, C, Y1, lambda_, mu)

        # 更新 Z
        Z = ALM_update_Z(C @ D ** (-1 / 2) @ C.t() + gamma * torch.eye(n), q, b, rho2, eta, max_iterations, epsilon)

        # 更新 alpha
        alpha = update_alpha(C, high_order_connection_matrices, theta)

        # 更新 Y1, Y2, 和 μ
        Y1 = Y1 + mu * (CA_matrix - C - E)
        Y2 = Y2 + mu * (C - J)
        mu = min(rho1 * mu, mu_max)

        # 检查收敛条件
        if torch.norm(CA_matrix - C - E, p=torch.inf) < epsilon and torch.norm(C - J, p=torch.inf) < epsilon:
            break

    return C, E, Z

# AWEC - H 实现
def AWEC_H(CA_matrix, lambda_, gamma, alpha, Y1, Y2, mu, max_iterations, epsilon, high_order_connection_matrices, theta, rho1, mu_max, q, b, rho2, eta, n_clusters):
    """
    AWEC - H 实现，使用平均链接层次凝聚聚类方法。

    params: 
    CA_matrix (torch.Tensor): 邻域矩阵。
    lambda_ (float): 正则化参数。
    gamma (float): 参数。
    alpha (list of float): 权重列表。
    Y1 (torch.Tensor): 拉格朗日乘子矩阵。
    Y2 (torch.Tensor): 拉格朗日乘子矩阵。
    mu (float): 惩罚参数。
    max_iterations (int): 最大迭代次数。
    epsilon (float): 收敛阈值。
    high_order_connection_matrices (list of torch.Tensor): 高阶连接矩阵列表。
    theta (float): 参数。
    rho1 (float): 参数。
    mu_max (float): 惩罚参数最大值。
    q (torch.Tensor): 辅助向量。
    b (torch.Tensor): 辅助向量。
    rho2 (float): 参数。
    eta (float): 参数。
    n_clusters (int): 聚类数。

    return: 
    np.ndarray: 聚类标签。
    """
    C, E, Z = AWEC(CA_matrix, lambda_, gamma, alpha, Y1, Y2, mu, max_iterations, epsilon, high_order_connection_matrices, theta, rho1, mu_max, q, b, rho2, eta)
    # 使用平均链接层次凝聚聚类方法
    Z_np = Z.numpy()
    if Z_np.shape[1] == 0:
        raise ValueError("Z matrix has 0 features after conversion to numpy array.")
    # 处理全为 NaN 的列
    Z_np[np.isnan(Z_np)] = 0
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    Z_np = imputer.fit_transform(Z_np)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    labels = clustering.fit_predict(Z_np)
    return labels

# AWEC - S 实现
def AWEC_S(CA_matrix, lambda_, gamma, alpha, Y1, Y2, mu, max_iterations, epsilon, high_order_connection_matrices, theta, rho1, mu_max, q, b, rho2, eta, n_clusters):
    """
    AWEC - S 实现，使用谱聚类方法。

    param
    CA_matrix (torch.Tensor): 邻域矩阵。
    lambda_ (float): 正则化参数。
    gamma (float): 参数。
    alpha (list of float): 权重列表。
    Y1 (torch.Tensor): 拉格朗日乘子矩阵。
    Y2 (torch.Tensor): 拉格朗日乘子矩阵。
    mu (float): 惩罚参数。
    max_iterations (int): 最大迭代次数。
    epsilon (float): 收敛阈值。
    high_order_connection_matrices (list of torch.Tensor): 高阶连接矩阵列表。
    theta (float): 参数。
    rho1 (float): 参数。
    mu_max (float): 惩罚参数最大值。
    q (torch.Tensor): 辅助向量。
    b (torch.Tensor): 辅助向量。
    rho2 (float): 参数。
    eta (float): 参数。
    n_clusters (int): 聚类数。

    return: 
    np.ndarray: 聚类标签。
    """
    C, E, Z = AWEC(CA_matrix, lambda_, gamma, alpha, Y1, Y2, mu, max_iterations, epsilon, high_order_connection_matrices, theta, rho1, mu_max, q, b, rho2, eta)
    # 使用谱聚类方法
    from sklearn.cluster import SpectralClustering
    Z_np = Z.numpy()
    if Z_np.shape[1] == 0:
        raise ValueError("Z matrix has 0 features after conversion to numpy array.")
    # 处理全为 NaN 的列
    Z_np[np.isnan(Z_np)] = 0
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    Z_np = imputer.fit_transform(Z_np)
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    labels = clustering.fit_predict(Z_np)
    return labels

# 测试代码
if __name__ == '__main__':
    ca_matrix = torch.randint(0, 2, (10, 10))  # 示例 CA 矩阵
    kernel_function = lambda i, j: 1 if i != j else 0  # 示例核函数
    order = 3  # 示例阶数
    high_order_connection_matrices = [build_high_order_connection_matrix(calculate_first_order_connectivity(ca_matrix, kernel_function), i + 1) for i in range(order)]
    alpha = [0.3, 0.3, 0.4]  # 示例权重列表
    lambda_ = 0.1
    gamma = 0.5
    Y1 = torch.rand((10, 10))
    Y2 = torch.rand((10, 10))
    mu = 0.1
    rho1 = 1.2
    mu_max = 108
    q = torch.rand(10)
    b = torch.rand(10)
    rho2 = 1.5
    eta = 10
    max_iterations = 100
    epsilon = 1e-6
    n_clusters = 3  # 假设聚类数为 3
    theta = 0.1  # 示例 theta 值

    # 进行 AWEC - H 聚类
    labels_h = AWEC_H(ca_matrix, lambda_, gamma, alpha, Y1, Y2, mu, max_iterations, epsilon, high_order_connection_matrices, theta, rho1, mu_max, q, b, rho2, eta, n_clusters)
    print("AWEC - H Labels:", labels_h)

    # 进行 AWEC - S 聚类
    labels_s = AWEC_S(ca_matrix, lambda_, gamma, alpha, Y1, Y2, mu, max_iterations, epsilon, high_order_connection_matrices, theta, rho1, mu_max, q, b, rho2, eta, n_clusters)
    print("AWEC - S Labels:", labels_s)
