import numpy as np
from typing import Union, Optional, Any, List

class GDA:
    def __init__(self, shared_cov: bool = True) -> None:
        """Gaussian Discriminant Analysis."""
        self.shared_cov = shared_cov
        
    def fit(self, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any]) -> "GDA":
        """Fit the model to data."""
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
            
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        m, n = X.shape
        
        self.priors = {}
        self.means = {}
        
        if self.shared_cov:
            cov = np.zeros((n, n))
            
        for k in self.classes:
            X_k = X[y == k]
            m_k = len(X_k)
            
            self.priors[k] = m_k / m
            self.means[k] = np.mean(X_k, axis=0)
            
            if self.shared_cov:
                diff = X_k - self.means[k]
                cov += diff.T @ diff
            else:
                if not hasattr(self, 'covs'):
                    self.covs = {}
                cov_k = np.cov(X_k, rowvar=False, bias=True)
                # Add small regularization to prevent singularity
                cov_k += np.eye(n) * 1e-6
                self.covs[k] = cov_k
                
        if self.shared_cov:
            cov = cov / m
            cov += np.eye(n) * 1e-6
            self.cov = cov
            
        return self

    def _log_pdf(self, X, mean, cov):
        """Multivariate normal log PDF."""
        d = X.shape[1]
        diff = X - mean
        
        sign, logdet = np.linalg.slogdet(cov)
        cov_inv = np.linalg.pinv(cov)
        
        # Calculate mahalanobis distance
        md = np.sum(diff @ cov_inv * diff, axis=1)
        
        log_pdf = -0.5 * (d * np.log(2 * np.pi) + logdet + md)
        return log_pdf
        
    def predict_proba(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Return class probabilities."""
        X = np.asarray(X)
        log_posteriors = []
        
        for k in self.classes:
            cov = self.cov if self.shared_cov else self.covs[k]
            log_prior = np.log(self.priors[k])
            log_lik = self._log_pdf(X, self.means[k], cov)
            log_posteriors.append(log_lik + log_prior)
            
        # Log-sum-exp trick for numerical stability
        log_posteriors = np.column_stack(log_posteriors) # type: ignore
        max_log = np.max(log_posteriors, axis=1, keepdims=True)
        exp_vals = np.exp(log_posteriors - max_log)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return probs
        
    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes[idx]
