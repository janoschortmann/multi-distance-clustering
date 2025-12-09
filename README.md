# multi-distance-clustering

A Python package for clustering using **two distance matrices**, supporting:

- Weighted distance fusion:  
  \
  D = α D₁ + (1 − α) D₂
  \
- Automatic selection of the weight α via **stability maximization**
- Multiple clustering backends:
  - k-medoids (via `scikit-learn-extra`)
  - spectral clustering
  - multi-kernel spectral clustering (via `mvlearn`)
  - EAC consensus clustering (via `eac`)
- Silhouette evaluation for precomputed distances
- Distance matrix heatmaps and dendrogram plots
- Modular, extensible, object-oriented implementation

---

##  Installation

Clone the repository and install in editable mode:

```bash
pip install -e .
```

This automatically installs required dependencies:
```
numpy, scipy, scikit-learn, scikit-learn-extra, joblib,
matplotlib, seaborn, mvlearn, eac
```

---

##  Example usage

```python
import numpy as np
from sklearn.metrics import pairwise_distances
from multi_distance_clustering import MultiDistanceClustering

# Generate synthetic data
X = np.random.randn(50, 3)
D1 = pairwise_distances(X, metric="euclidean")
D2 = pairwise_distances(X, metric="manhattan")

# Create clustering object
clu = MultiDistanceClustering(D1, D2, n_clusters=3, method="kmedoids")

# Automatic alpha selection
alpha, stability, results = clu.select_alpha()
print("Selected alpha:", alpha)

# Fit clustering model
labels = clu.fit_predict()
print("Cluster labels:", labels)

# Evaluate quality
print("Silhouette score:", clu.silhouette())

# Plot results
clu.plot_distances()
clu.plot_weighted_dendrogram()
```

---

##  Features

### Automatic alpha selection  
Alpha is chosen by maximizing clustering **stability** over random subsamples using ARI.

### Spectral & multi-kernel clustering  
Support for multi-view spectral clustering using mvlearn.

### Consensus clustering (EAC)  
Combine two distance-based clusterings into a consensus partition.

### Visualization utilities  
Heatmaps and dendrograms for weighted and input distances.

---

## Package structure

```
multi_distance_clustering/
│
├── clustering/
│   └── multi_distance.py
├── stability/
│   └── stability.py
├── utils/
│   ├── distances.py
│   └── plotting.py
└── __init__.py
```

---

## Running the tests

Once installed:

```bash
pytest
```

Tests are included in the `tests/` directory.

---

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.


