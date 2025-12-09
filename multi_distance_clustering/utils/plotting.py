import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


class PlottingMixin:
    """Mixin used by clustering classes to add plotting functionality."""

    def plot_heatmap(self, D: np.ndarray, title: str = "") -> None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(D, cmap="viridis")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_dendrogram(self, D: np.ndarray, title: str = "") -> None:
        Z = linkage(squareform(D), method="average")
        plt.figure(figsize=(8, 4))
        dendrogram(Z)
        plt.title(title)
        plt.tight_layout()
        plt.show()
