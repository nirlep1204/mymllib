import numpy as np
import pandas as pd
from typing import Optional, Any, Union

class PCA:
    """Principal Component Analysis."""
    
    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance = None
        self._all_eigenvalues_sum = None

    def fit(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> "PCA":
        """Fit the PCA model on X."""
        X_arr = np.asarray(X, dtype=float)
        self.mean = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean
        
        m = X_centered.shape[0]
        cov = (X_centered.T @ X_centered) / m
        
        vals, vecs = np.linalg.eigh(cov)
        
        # Sort by eigenvalue in descending order
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        self._all_eigenvalues_sum = np.sum(vals)
        
        # Keep top n_components
        self.eigenvalues = vals[:self.n_components]
        self.components = vecs[:, :self.n_components].T  # rows are principal directions
        self.explained_variance = self.eigenvalues
        
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Apply dimensionality reduction to X."""
        X_arr = np.asarray(X, dtype=float)
        X_centered = X_arr - self.mean
        X_proj = X_centered @ self.components.T
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_proj, index=X.index, columns=[f"PC{i+1}" for i in range(self.n_components)])
        return X_proj

    def fit_transform(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Fit model and apply transform to X."""
        return self.fit(X).transform(X)

    def reconstruct(self, X_proj: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Transform data back to its original space."""
        X_proj_arr = np.asarray(X_proj, dtype=float)
        X_rec = X_proj_arr @ self.components + self.mean
        
        if isinstance(X_proj, pd.DataFrame):
            return pd.DataFrame(X_rec, index=X_proj.index)
        return X_rec

    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Ratio of variance explained by each component."""
        if self.eigenvalues is None:
            return None
        return self.eigenvalues / self._all_eigenvalues_sum
