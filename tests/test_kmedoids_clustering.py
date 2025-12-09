import numpy as np
from sklearn.metrics import pairwise_distances
from multi_distance_clustering import MultiDistanceClustering


def test_kmedoids_fit_predict():
    X = np.random.randn(20, 3)
    D = pairwise_distances(X)

    clu = MultiDistanceClustering(D, D, n_clusters=3, method="kmedoids")
    clu.fit_predict(alpha=0.5)

    assert clu.labels_ is not None
    assert len(clu.labels_) == D.shape[0]
