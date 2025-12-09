import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.sparse.csgraph import laplacian


def rbf_kernel_from_distance(D, gamma=None):
    """
    Convert a distance matrix to an RBF kernel.
    gamma = 1/(2*sigma^2). If None, gamma = 1/median(D)^2.
    """
    D = np.asarray(D)
    if gamma is None:
        # Typical heuristic: sigma = median(D); gamma = 1/(2 sigma^2)
        med = np.median(D[D > 0])
        gamma = 1.0 / (2.0 * med * med)
    return np.exp(-gamma * D * D)


def multi_kernel_spectral_clustering(
    D1,
    D2,
    n_clusters,
    alpha=0.5,
    gamma1=None,
    gamma2=None,
    random_state=42,
):
    """
    Multi-kernel spectral clustering using RBF kernels from two distances.

    Parameters
    ----------
    D1, D2 : ndarray
        Distance matrices (n x n).
    n_clusters : int
        Number of clusters to find.
    alpha : float
        Weight for first kernel K1; K = alpha*K1 + (1-alpha)*K2
    gamma1, gamma2 : float or None
        RBF scale parameters for the kernels.
    random_state : int
        Random seed for k-means.

    Returns
    -------
    labels : ndarray of shape (n,)
    """
    # Build kernels
    K1 = rbf_kernel_from_distance(D1, gamma=gamma1)
    K2 = rbf_kernel_from_distance(D2, gamma=gamma2)

    # Fuse kernels
    K = alpha * K1 + (1.0 - alpha) * K2

    # Normalized Laplacian
    L = laplacian(K, normed=True)

    # Eigen-decomposition: we take smallest eigenvalues
    eigvals, eigvecs = np.linalg.eigh(L)

    idx = np.argsort(eigvals)[:n_clusters]
    U = eigvecs[:, idx]   # n x k matrix

    # Row-normalize eigenvectors (common in spectral clustering)
    U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)

    # Cluster via k-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(U_norm)

    return labels
