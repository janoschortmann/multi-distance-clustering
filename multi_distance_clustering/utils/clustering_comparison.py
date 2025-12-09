import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


class ClusteringComparison:
    """
    Utility to compare multiple clusterings produced on the same dataset.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n_samples, n_samples)
        Distance matrix used for silhouette scores.
    clusterings : dict
        A mapping: method_name -> labels_array
    """

    def __init__(self, distance_matrix, clusterings):
        self.D = np.asarray(distance_matrix)
        self.clusterings = clusterings  # dict: name -> labels
        self.methods = list(clusterings.keys())

    # ---------------------------------------------------------
    # Pairwise metrics
    # ---------------------------------------------------------
    def pairwise_ari(self):
        """Return a matrix of pairwise Adjusted Rand Index."""
        m = len(self.methods)
        M = np.zeros((m, m))
        for i, a in enumerate(self.methods):
            for j, b in enumerate(self.methods):
                M[i, j] = adjusted_rand_score(
                    self.clusterings[a], self.clusterings[b]
                )
        return M

    def pairwise_nmi(self):
        """Return a matrix of pairwise Normalized Mutual Information."""
        m = len(self.methods)
        M = np.zeros((m, m))
        for i, a in enumerate(self.methods):
            for j, b in enumerate(self.methods):
                M[i, j] = normalized_mutual_info_score(
                    self.clusterings[a], self.clusterings[b]
                )
        return M

    # ---------------------------------------------------------
    # Individual quality metrics
    # ---------------------------------------------------------
    def silhouette_scores(self):
        """Return dict: method -> silhouette score."""
        return {
            name: silhouette_score(self.D, labels, metric="precomputed")
            for name, labels in self.clusterings.items()
        }

    # ---------------------------------------------------------
    # Plotting utilities
    # ---------------------------------------------------------
    def plot_pairwise_metric(self, metric="ari", figsize=(6, 5), cmap="viridis"):
        """
        Plot ARI or NMI heatmap.
        """
        if metric == "ari":
            M = self.pairwise_ari()
            title = "Pairwise Adjusted Rand Index"
        elif metric == "nmi":
            M = self.pairwise_nmi()
            title = "Pairwise Normalized Mutual Information"
        else:
            raise ValueError("metric must be 'ari' or 'nmi'.")

        plt.figure(figsize=figsize)
        sns.heatmap(
            M,
            xticklabels=self.methods,
            yticklabels=self.methods,
            annot=True,
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_silhouette_bars(self, figsize=(6, 4)):
        """Plot silhouette score for each clustering."""
        scores = self.silhouette_scores()

        plt.figure(figsize=figsize)
        sns.barplot(
            x=list(scores.keys()),
            y=list(scores.values()),
            palette="viridis",
        )
        plt.ylim(0, 1)
        plt.ylabel("Silhouette score")
        plt.title("Silhouette Score Comparison")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    def summary(self):
        """
        Return a dict containing:
        - silhouette scores
        - pairwise ARI matrix
        - pairwise NMI matrix
        """
        return {
            "silhouette": self.silhouette_scores(),
            "ari": self.pairwise_ari(),
            "nmi": self.pairwise_nmi(),
            "methods": self.methods,
        }
