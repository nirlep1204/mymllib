import numpy as np
import pandas as pd
from typing import Optional, Union, List, Any

class GMM:
    """Gaussian Mixture Model using EM algorithm."""
    
    def __init__(self, k: int = 3, max_iter: int = 100, tol: float = 1e-6, seed: Optional[int] = None) -> None:
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.weights: Any = None
        self.means: Any = None
        self.covariances: Any = None
        self.labels: Any = None
        self.losses: List[Any] = []
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> "GMM":
        """Fit the model to the data."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_arr = X.values
        else:
            X_arr = np.array(X)
            
        n_samples, n_features = X_arr.shape # type: ignore
        
        if self.seed is not None:
            np.random.seed(self.seed)
            
        self.weights = np.ones(self.k) / self.k
        
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.means = X_arr[idx].copy()
        
        self.covariances = np.array([np.eye(n_features) for _ in range(self.k)])
        
        prev_log_likelihood = -np.inf
        self.losses = []
        
        for _ in range(self.max_iter):
            # E-step
            log_resp = np.zeros((n_samples, self.k))
            
            for j in range(self.k):
                log_pdf = self._gaussian_log_pdf(X_arr, self.means[j], self.covariances[j])
                log_resp[:, j] = np.log(self.weights[j]) + log_pdf
                
            # Log-sum-exp trick for numerical stability
            max_log_resp = np.max(log_resp, axis=1, keepdims=True)
            sum_exp = np.sum(np.exp(log_resp - max_log_resp), axis=1, keepdims=True)
            log_likelihoods = max_log_resp + np.log(sum_exp)
            
            resp = np.exp(log_resp - log_likelihoods)
            
            current_log_likelihood = np.sum(log_likelihoods)
            self.losses.append(current_log_likelihood)
            
            if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = current_log_likelihood
            
            # M-step
            N_k = np.sum(resp, axis=0)
            
            for j in range(self.k):
                # Handle collapsing components
                if N_k[j] < 1e-8:
                    self.weights[j] = 1.0 / self.k
                    self.means[j] = X_arr[np.random.choice(n_samples)]
                    self.covariances[j] = np.eye(n_features)
                    continue
                    
                self.weights[j] = N_k[j] / n_samples
                self.means[j] = np.sum(resp[:, j:j+1] * X_arr, axis=0) / N_k[j]
                
                diff = X_arr - self.means[j]
                cov = (resp[:, j:j+1] * diff).T @ diff / N_k[j]
                self.covariances[j] = cov + 1e-6 * np.eye(n_features)
                
        # Store final hard assignments
        self.labels = np.argmax(self._get_resp(X_arr), axis=1)
        
        return self
        
    def _get_resp(self, X):
        """Compute responsibilities matrix."""
        n_samples = X.shape[0]
        log_resp = np.zeros((n_samples, self.k))
        
        for j in range(self.k):
            log_pdf = self._gaussian_log_pdf(X, self.means[j], self.covariances[j])
            log_resp[:, j] = np.log(self.weights[j]) + log_pdf
            
        max_log_resp = np.max(log_resp, axis=1, keepdims=True)
        sum_exp = np.sum(np.exp(log_resp - max_log_resp), axis=1, keepdims=True)
        log_likelihoods = max_log_resp + np.log(sum_exp)
        
        return np.exp(log_resp - log_likelihoods)
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """Return hard cluster assignments."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_arr = X.values
        else:
            X_arr = np.array(X)
            
        resp = self._get_resp(X_arr)
        labels = np.argmax(resp, axis=1)
        
        if isinstance(X, pd.DataFrame):
            return pd.Series(labels, index=X.index, name="cluster")
        return labels
        
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Union[np.ndarray, pd.DataFrame]:
        """Return responsibilities matrix (soft assignments)."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_arr = X.values
        else:
            X_arr = np.array(X)
            
        resp = self._get_resp(X_arr)
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(resp, index=X.index, columns=[f"cluster_{i}" for i in range(self.k)])
        return resp
        
    def _gaussian_log_pdf(self, X, mean, cov):
        """Compute log multivariate Gaussian PDF."""
        n_features = X.shape[1]
        
        sign, log_det = np.linalg.slogdet(cov)
        if log_det < -1e10 or np.isinf(log_det) or sign <= 0:
            inv_cov = np.linalg.pinv(cov)
            u, s, vh = np.linalg.svd(cov)
            log_det = np.sum(np.log(s[s > 1e-10]))
        else:
            inv_cov = np.linalg.inv(cov)
            
        diff = X - mean
        mahalanobis = np.sum((diff @ inv_cov) * diff, axis=1)
        
        log_pdf = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis)
        return log_pdf
