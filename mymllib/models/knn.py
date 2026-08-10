import numpy as np
from typing import Union, Any

class KNN:
    def __init__(self, k: int = 5, task: str = 'classify') -> None:
        """K-Nearest Neighbors."""
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        if task not in ('classify', 'regress'):
            raise ValueError("task must be 'classify' or 'regress'.")
            
        self.k = k
        self.task = task
        
    def fit(self, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any]) -> "KNN":
        """Store the training data."""
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).ravel()
        
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
            
        return self
        
    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Predict labels or values for X."""
        X = np.asarray(X)
        
        # Efficient distance calculation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        train_sq = np.sum(self.X_train**2, axis=1)
        dot = X @ self.X_train.T
        
        dists = X_sq + train_sq - 2 * dot
        
        n_samples = X.shape[0]
        preds = np.zeros(n_samples, dtype=self.y_train.dtype)
        
        # Ensure we don't ask for more neighbors than we have
        k = min(self.k, self.X_train.shape[0])
        
        # Find indices of k smallest distances using argpartition (O(n))
        nn_idx = np.argpartition(dists, k - 1, axis=1)[:, :k]
        
        for i in range(n_samples):
            k_nearest_y = self.y_train[nn_idx[i]]
            
            if self.task == 'classify':
                labels, counts = np.unique(k_nearest_y, return_counts=True)
                max_count = np.max(counts)
                
                # Tie breaking: pick smallest label among those with max count
                candidates = labels[counts == max_count]
                preds[i] = np.min(candidates)
            else:
                preds[i] = np.mean(k_nearest_y)
                
        return preds
