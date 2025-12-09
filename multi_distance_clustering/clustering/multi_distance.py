import numpy as np
from joblib import Parallel, delayed

from sklearn_extra.cluster import KMedoids
from sklearn.cluster import SpectralClustering

from ..utils.distances import DistanceUtils
from ..utils.plotting import PlottingMixin
from ..stability.stability import StabilityEvaluator
from ..utils.eac import eac_consensus
from ..utils.multi_kernel_spectral import multi_kernel_spectral_clustering



class MultiDistanceClustering(PlottingMixin):
    """
    Clustering with two distance matrices, supporting:

    - "kmedoids"              : k-medoids on weighted distance
    - "spectral"              : spectral clustering on RBF kernel of weighted distance
    - "multi_kernel_spectral" : multi-view spectral clustering
    - "eac"                   : Evidence Accumulation Clustering (eac)

    Weighted distance: D = alpha * D1 + (1 - alpha) * D2.
    """

    SUPPORTED_METHODS = {
        "kmedoids",
        "spectral",
        "multi_kernel_spectral",
        "eac",
    }

    def __init__(
        self,
        D1,
        D2,
        n_clusters=3,
        method="kmedoids",
        random_state=42,
        stability_n_runs=10,
        stability_subsample=0.8,
    ):
        self.D1 = np.asarray(D1)
        self.D2 = np.asarray(D2)

        if self.D1.shape != self.D2.shape:
            raise ValueError("D1 and D2 must have same shape.")

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {method}")

        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.alpha = None
        self.labels_ = None

        self.stability_eval = StabilityEvaluator(
            n_runs=stability_n_runs,
            subsample=stability_subsample,
            random_state=random_state if random_state is not None else 42,
        )

    # ---------------------------------------------------------
    # Weighted distance
    # ---------------------------------------------------------
    def weighted_distance(self, alpha):
        return alpha * self.D1 + (1 - alpha) * self.D2

    # ---------------------------------------------------------
    # Clustering backends
    # ---------------------------------------------------------
    def _fit_kmedoids(self, D):
        return KMedoids(
            n_clusters=self.n_clusters,
            metric="precomputed",
            init="k-medoids++",
            random_state=self.random_state,
        ).fit_predict(D)

    def _fit_spectral(self, D):
        K = DistanceUtils.rbf_kernel(D)
        return SpectralClustering(
            n_clusters=self.n_clusters,
            affinity="precomputed",
            random_state=self.random_state,
        ).fit_predict(K)

    def _fit_multi_kernel_spectral(self, D1, D2):
        """
        Internal multi-kernel spectral clustering without mvlearn.
        Uses RBF kernels and fuses them with current alpha.
        """
        alpha = self.alpha if self.alpha is not None else 0.5

        labels = multi_kernel_spectral_clustering(
            D1,
            D2,
            n_clusters=self.n_clusters,
            alpha=alpha,
        )
        return labels
    

    def _fit_eac(self, D1, D2):
        """
        EAC consensus of two base k-medoids clusterings:
        - one from D1
        - one from D2

        Currently both are weighted equally (0.5, 0.5) in the
        co-association matrix.
        """
        labels1 = self._fit_kmedoids(D1)
        labels2 = self._fit_kmedoids(D2)

        # Equal weights; if you later want to encode "confidence" of D1,
        # you can use weights=[alpha, 1-alpha] instead.
        labels_eac, _ = eac_consensus(
            [labels1, labels2],
            n_clusters=self.n_clusters,
            weights=[0.5, 0.5],
        )
        return labels_eac
    

    # ---------------------------------------------------------
    # Cluster fun wrapper for stability (only for kmedoids/spectral)
    # ---------------------------------------------------------
    def _cluster_fun(self, alpha):
        def cluster_on_submatrix(Dsub):
            if self.method == "kmedoids":
                return self._fit_kmedoids(Dsub)
            if self.method == "spectral":
                return self._fit_spectral(Dsub)
            raise ValueError(
                "Stability-based alpha search only implemented "
                "for 'kmedoids' and 'spectral'."
            )

        return cluster_on_submatrix

    # ---------------------------------------------------------
    # Alpha selection
    # ---------------------------------------------------------
    def select_alpha(self, alphas=np.linspace(0, 1, 21), n_jobs=-1):
        """
        Select alpha via stability (ARI on subsamples).

        Only meaningful for methods: 'kmedoids' and 'spectral'.
        """

        if self.method not in {"kmedoids", "spectral"}:
            raise ValueError(
                "select_alpha() only supports 'kmedoids' and 'spectral'."
            )

        def eval_one(a):
            D = self.weighted_distance(a)
            score = self.stability_eval.compute(D, self._cluster_fun(a))
            return a, score

        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_one)(a) for a in alphas
        )

        self.alpha, best_score = max(results, key=lambda x: x[1])
        return self.alpha, best_score, results

    # ---------------------------------------------------------
    # Fit / predict
    # ---------------------------------------------------------
    def fit(self, alpha=None):
        if self.method in {"kmedoids", "spectral"}:
            if alpha is not None:
                self.alpha = alpha
            if self.alpha is None:
                raise ValueError(
                    "Call select_alpha() or pass alpha explicitly to fit()."
                )
            Dw = self.weighted_distance(self.alpha)
            if self.method == "kmedoids":
                self.labels_ = self._fit_kmedoids(Dw)
            else:
                self.labels_ = self._fit_spectral(Dw)

        elif self.method == "multi_kernel_spectral":
            self.labels_ = self._fit_multi_kernel_spectral(self.D1, self.D2)

        elif self.method == "eac":
            self.labels_ = self._fit_eac(self.D1, self.D2)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def fit_predict(self, alpha=None):
        return self.fit(alpha).labels_

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    def silhouette(self):
        if self.labels_ is None:
            raise ValueError("Call fit() first.")
        if self.alpha is None and self.method in {"kmedoids", "spectral"}:
            raise ValueError("alpha is not set.")
        # For multi_kernel_spectral or eac, we can still use weighted D with alpha=0.5
        if self.method in {"multi_kernel_spectral", "eac"} and self.alpha is None:
            alpha = 0.5
        else:
            alpha = self.alpha
        D = self.weighted_distance(alpha)
        return DistanceUtils.silhouette(D, self.labels_)
