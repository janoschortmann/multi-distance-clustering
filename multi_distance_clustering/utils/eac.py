import numpy as np
from typing import Sequence, Optional, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def eac_coassociation(
    labelings: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Compute the EAC co-association matrix from a list of labelings.

    Parameters
    ----------
    labelings : sequence of 1D int arrays
        Each array is of shape (n_samples,) and contains cluster labels
        for the same set of points.
    weights : sequence of float, optional
        Non-negative weights for each labeling. If None, all weights are 1.

    Returns
    -------
    C : ndarray of shape (n_samples, n_samples)
        Co-association matrix with entries in [0, 1], symmetric, with
        diagonal entries equal to 1.
    """
    if len(labelings) == 0:
        raise ValueError("labelings must be a non-empty sequence.")

    n = len(labelings[0])
    for lab in labelings:
        if len(lab) != n:
            raise ValueError("All labelings must have the same length.")

    m = len(labelings)
    if weights is None:
        weights = np.ones(m, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != m:
            raise ValueError("weights must have same length as labelings.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")

    C = np.zeros((n, n), dtype=float)
    total_weight = float(np.sum(weights))

    if total_weight == 0:
        raise ValueError("Sum of weights must be positive.")

    # Accumulate co-occurrences
    for lab, w in zip(labelings, weights):
        if w == 0:
            continue
        lab = np.asarray(lab)
        # Fast cluster-by-cluster update
        for cluster_id in np.unique(lab):
            idx = np.where(lab == cluster_id)[0]
            # Add w to all pairs within the cluster
            C[np.ix_(idx, idx)] += w

    # Normalize to [0,1] co-association
    C /= total_weight
    # Ensure symmetry and ones on diagonal
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 1.0)
    return C


def eac_consensus(
    labelings: Sequence[np.ndarray],
    n_clusters: int,
    weights: Optional[Sequence[float]] = None,
    linkage_method: str = "average",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform EAC consensus clustering.

    Parameters
    ----------
    labelings : sequence of 1D int arrays
        Base clusterings.
    n_clusters : int
        Desired number of clusters in the final partition.
    weights : sequence of float, optional
        Weights for each labeling.
    linkage_method : {"single", "complete", "average", "ward"}, default="average"
        Linkage method used by SciPy's linkage.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Consensus cluster labels.
    C : ndarray of shape (n_samples, n_samples)
        Co-association matrix used for clustering.
    """
    C = eac_coassociation(labelings, weights=weights)
    # Convert similarity to distance
    D = 1.0 - C
    # SciPy linkage expects condensed distance
    Z = linkage(squareform(D), method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels, C
