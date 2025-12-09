import numpy as np
from sklearn.metrics import pairwise_distances
from multi_distance_clustering import MultiDistanceClustering


def test_multi_kernel():
    X = np.random.randn(20, 3)
    D1 = pairwise_distances(X)
    D2 = pairwise_distances(X)

    clu = MultiDistanceClustering(D1, D2, n_clusters=3, method="multi_kernel_spectral")
    labels = clu.fit_predict()
    assert len(labels) == len(X)


def test_eac():
    X = np.random.randn(20, 3)
    D1 = pairwise_distances(X)
    D2 = pairwise_distances(X)

    clu = MultiDistanceClustering(D1, D2, n_clusters=3, method="eac")
    labels = clu.fit_predict()

    assert len(labels) == len(X)
