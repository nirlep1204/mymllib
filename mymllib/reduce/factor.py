import numpy as np
import pandas as pd
from typing import Optional, Any, Union

class Factor:
    """Factor Analysis using EM algorithm."""
    
    def __init__(self, n_factors: int = 2, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.tol = tol
        self.mean = None
        self._loadings = None
        self.noise = None

    def fit(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> "Factor":
        """Fit Factor Analysis model on X."""
        X_arr = np.asarray(X, dtype=float)
        self.mean = np.mean(X_arr, axis=0)
        X_c = X_arr - self.mean
        m, n = X_c.shape
        
        S = (X_c.T @ X_c) / m
        
        vals, vecs = np.linalg.eigh(S)
        idx = np.argsort(vals)[::-1][:self.n_factors]
        self._loadings = vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0))
        self.noise = np.diag(S).copy() # type: ignore
        
        for _ in range(self.max_iter):
            L = self._loadings
            Psi_inv = np.diag(1.0 / (self.noise + 1e-10)) # type: ignore
            
            # E-step
            M = L.T @ Psi_inv @ L + np.eye(self.n_factors) # type: ignore
            M_inv = np.linalg.inv(M)
            
            beta = M_inv @ L.T @ Psi_inv # type: ignore
            E_z = X_c @ beta.T
            
            E_zz = m * M_inv + E_z.T @ E_z
            
            # M-step
            L_new = (X_c.T @ E_z) @ np.linalg.inv(E_zz)
            Psi_new = np.diag(S) - np.diag(L_new @ (E_z.T @ X_c) / m)
            Psi_new = np.maximum(Psi_new, 1e-10)
            
            # Check convergence
            diff = np.max(np.abs(L_new - L))
            self._loadings = L_new
            self.noise = Psi_new
            
            if diff < self.tol:
                break
                
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Transform X to the latent space."""
        X_arr = np.asarray(X, dtype=float)
        X_c = X_arr - self.mean
        
        Psi_inv = np.diag(1.0 / (self.noise + 1e-10)) # type: ignore
        M = self._loadings.T @ Psi_inv @ self._loadings + np.eye(self.n_factors) # type: ignore
        M_inv = np.linalg.inv(M)
        
        z = X_c @ Psi_inv @ self._loadings @ M_inv
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(z, index=X.index, columns=[f"Factor{i+1}" for i in range(self.n_factors)])
        return z

    def fit_transform(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Fit model and apply transform to X."""
        return self.fit(X).transform(X)

    @property
    def loadings(self) -> np.ndarray:
        """Factor loadings matrix."""
        return self._loadings # type: ignore
