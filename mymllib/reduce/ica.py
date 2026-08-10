import numpy as np
import pandas as pd

class ICA:
    """Independent Component Analysis using FastICA."""
    
    def __init__(self, n_components=2, max_iter=200, tol=1e-4, seed=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.unmixing = None
        self.mean = None
        self.whitening = None

    def _g(self, u):
        """tanh non-linearity."""
        return np.tanh(u)
        
    def _g_prime(self, u):
        """Derivative of tanh."""
        return 1.0 - np.tanh(u)**2

    def fit(self, X):
        """Fit the ICA model on X."""
        if self.seed is not None:
            np.random.seed(self.seed)
            
        X_arr = np.asarray(X, dtype=float)
        m, n = X_arr.shape
        self.mean = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean
        
        # Whitening
        cov = (X_centered.T @ X_centered) / m
        vals, vecs = np.linalg.eigh(cov)
        
        idx = np.argsort(vals)[::-1]
        vals = vals[idx][:self.n_components]
        vecs = vecs[:, idx][:, :self.n_components]
        
        D = np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-10)))
        self.whitening = D @ vecs.T
        X_white = X_centered @ self.whitening.T
        
        # FastICA with deflation
        W = np.zeros((self.n_components, self.n_components))
        
        for p in range(self.n_components):
            w = np.random.randn(self.n_components)
            w = w / np.linalg.norm(w)
            
            for _ in range(self.max_iter):
                u = X_white @ w
                g_u = self._g(u)
                g_prime_u = self._g_prime(u)
                
                w_new = (X_white.T @ g_u) / m - np.mean(g_prime_u) * w
                
                # Orthogonalize against previous components
                if p > 0:
                    w_new = w_new - W[:p].T @ (W[:p] @ w_new)
                    
                w_new = w_new / np.linalg.norm(w_new)
                
                # Check convergence
                if np.abs(np.abs(np.dot(w_new, w)) - 1.0) < self.tol:
                    w = w_new
                    break
                    
                w = w_new
                
            W[p] = w
            
        self.unmixing = W
        return self

    def transform(self, X):
        """Recover independent components from X."""
        X_arr = np.asarray(X, dtype=float)
        X_centered = X_arr - self.mean
        X_white = X_centered @ self.whitening.T
        components = X_white @ self.unmixing.T
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(components, index=X.index, columns=[f"IC{i+1}" for i in range(self.n_components)])
        return components

    def fit_transform(self, X):
        """Fit model and apply transform to X."""
        return self.fit(X).transform(X)
