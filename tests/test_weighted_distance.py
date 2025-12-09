import numpy as np
from multi_distance_clustering import MultiDistanceClustering


def test_weighted_distance():
    D1 = np.array([[0, 1], [1, 0]])
    D2 = np.array([[0, 2], [2, 0]])

    clu = MultiDistanceClustering(D1, D2, n_clusters=2)
    Dw = clu.weighted_distance(alpha=0.25)

    expected = 0.25 * D1 + 0.75 * D2
    assert np.allclose(Dw, expected)
