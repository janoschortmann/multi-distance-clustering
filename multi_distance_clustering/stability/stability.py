from dataclasses import dataclass
import numpy as np
from sklearn.metrics import adjusted_rand_score


@dataclass
class StabilityEvaluator:
    n_runs: int = 10
    subsample: float = 0.8
    random_state: int = 42

    def compute(self, D, cluster_fun):
        rng = np.random.default_rng(self.random_state)
        n = D.shape[0]
        m = max(2, int(self.subsample * n))

        labels_list = []
        for _ in range(self.n_runs):
            idx = rng.choice(n, m, replace=False)
            D_sub = D[np.ix_(idx, idx)]
            labels_sub = cluster_fun(D_sub)
            labels_list.append((idx, labels_sub))

        scores = []
        for i in range(len(labels_list)):
            idx_i, lab_i = labels_list[i]
            for j in range(i + 1, len(labels_list)):
                idx_j, lab_j = labels_list[j]
                common, i_pos, j_pos = np.intersect1d(
                    idx_i, idx_j, return_indices=True
                )
                if len(common) > 1:
                    scores.append(adjusted_rand_score(lab_i[i_pos], lab_j[j_pos]))

        return float(np.mean(scores)) if scores else 0.0
