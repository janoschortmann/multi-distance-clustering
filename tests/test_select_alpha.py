import numpy as np
from sklearn.metrics import pairwise_distances
from multi_distance_clustering import MultiDistanceClustering


def test_select_alpha():
    X = np.random.randn(30, 3)
    D1 = pairwise_distances(X, metric="euclidean")
    D2 = pairwise_distances(X, metric="cosine")

    clu = MultiDistanceClustering(D1, D2, n_clusters=3, method="kmedoids")

    alpha, stability, grid = clu.select_alpha(alphas=[0.0, 0.5], n_jobs=1)

    assert 0.0 <= alpha <= 1.0
    assert isinstance(stability, float)
