import numpy as np
import pandas as pd
from typing import Optional, Union, List, Any

class KMeans:
    """K-Means clustering algorithm."""
    
    def __init__(self, k: int = 3, max_iter: int = 100, seed: Optional[int] = None) -> None:
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.centers: Any = None
        self.labels: Any = None
        self.losses: List[Any] = []

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "KMeans":
        """Fit the KMeans model to data."""
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy()
        else:
            X_arr = np.array(X)
            
        n_samples, n_features = X_arr.shape
        
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Initialize k centers randomly from data points
        idx = np.random.choice(n_samples, self.k, replace=(n_samples < self.k))
        self.centers = X_arr[idx].copy()
        
        for _ in range(self.max_iter):
            # Calculate vectorized distances to all centers
            distances = np.zeros((n_samples, self.k))
            for i in range(self.k):
                distances[:, i] = np.sum((X_arr - self.centers[i])**2, axis=1)
                
            # Assign points to nearest center
            new_labels = np.argmin(distances, axis=1)
            
            # Compute WCSS for current iteration
            wcss = 0
            for i in range(self.k):
                cluster_points = X_arr[new_labels == i]
                if len(cluster_points) > 0:
                    wcss += np.sum((cluster_points - self.centers[i])**2)
            self.losses.append(wcss)
            
            # Recompute centers
            new_centers = np.zeros_like(self.centers)
            for i in range(self.k):
                cluster_points = X_arr[new_labels == i]
                if len(cluster_points) == 0:
                    # Reinitialize empty cluster
                    new_centers[i] = X_arr[np.random.choice(n_samples)]
                else:
                    new_centers[i] = np.mean(cluster_points, axis=0)
                    
            # Check convergence
            if np.allclose(self.centers, new_centers):
                self.labels = new_labels
                break
                
            self.centers = new_centers
            self.labels = new_labels
            
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Assign new points to nearest center."""
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy()
        else:
            X_arr = np.array(X)
            
        distances = np.zeros((X_arr.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = np.sum((X_arr - self.centers[i])**2, axis=1)
            
        return np.argmin(distances, axis=1)

    def cost(self) -> float:
        """Return final within-cluster sum of squares (WCSS)."""
        return self.losses[-1] if self.losses else 0.0
