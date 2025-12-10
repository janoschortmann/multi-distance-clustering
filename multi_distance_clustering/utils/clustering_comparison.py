import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


from scipy.optimize import linear_sum_assignment

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
from scipy.optimize import linear_sum_assignment


    # ---------------------------------------------------------
    # Cluster overlap and optimal alignment
    # ---------------------------------------------------------
    def cluster_overlap_matrix(self, labels1, labels2):
        """
        Compute raw overlap counts between clusters of two labelings.

        Returns
        -------
        M : ndarray of shape (k1, k2)
            M[i, j] = number of points that are in cluster i of labels1
                      and cluster j of labels2.
        unique1, unique2 : arrays of unique labels
        """
        labels1 = np.asarray(labels1)
        labels2 = np.asarray(labels2)

        unique1 = np.unique(labels1)
        unique2 = np.unique(labels2)

        k1 = len(unique1)
        k2 = len(unique2)

        M = np.zeros((k1, k2), dtype=int)

        for i, c1 in enumerate(unique1):
            idx1 = np.where(labels1 == c1)[0]
            for j, c2 in enumerate(unique2):
                idx2 = np.where(labels2 == c2)[0]
                M[i, j] = len(np.intersect1d(idx1, idx2))

        return M, unique1, unique2


    def best_cluster_alignment(self, labels1, labels2):
        """
        Find best alignment between clusters of two labelings using Hungarian algorithm.

        Returns
        -------
        assignment : list of (cluster_from_labels1, cluster_from_labels2)
        M : overlap matrix
        """
        M, u1, u2 = self.cluster_overlap_matrix(labels1, labels2)

        # Hungarian algorithm wants a cost matrix to minimize → we maximize overlap
        cost = -M
        row_ind, col_ind = linear_sum_assignment(cost)

        alignment = [(u1[i], u2[j]) for i, j in zip(row_ind, col_ind)]
        return alignment, M, u1, u2


    def plot_overlap_heatmap(self, method1, method2, figsize=(6, 5)):
        """
        Plot heatmap of cluster overlap between two methods,
        after optimal alignment.
        """
        labels1 = self.clusterings[method1]
        labels2 = self.clusterings[method2]

        alignment, M, u1, u2 = self.best_cluster_alignment(labels1, labels2)

        plt.figure(figsize=figsize)
        sns.heatmap(M, annot=True, fmt="d",
                    xticklabels=[f"{method2}:{c}" for c in u2],
                    yticklabels=[f"{method1}:{c}" for c in u1],
                    cmap="Blues")
        plt.title(f"Cluster Overlap: {method1} vs {method2}\n(Optimal Matching)")
        plt.xlabel(method2)
        plt.ylabel(method1)
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


