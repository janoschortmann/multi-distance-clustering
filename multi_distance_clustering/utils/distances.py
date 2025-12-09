import numpy as np
from sklearn.metrics import silhouette_score


class DistanceUtils:
    """Utility functions for kernels, silhouettes, etc."""

    @staticmethod
    def rbf_kernel(D: np.ndarray, sigma=None) -> np.ndarray:
        D = np.asarray(D)
        if sigma is None:
            nonzero = D[D > 0]
            sigma = np.median(nonzero) if nonzero.size > 0 else 1.0
        return np.exp(-D ** 2 / (2 * sigma ** 2))

    @staticmethod
    def silhouette(D: np.ndarray, labels: np.ndarray) -> float:
        return silhouette_score(D, labels, metric="precomputed")