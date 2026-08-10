import numpy as np
import pandas as pd

class GLM:
    """Generalized Linear Model supporting Gaussian, Bernoulli, and Poisson families."""
    def __init__(self, family='gaussian', lr=0.01, max_iter=1000):
        if family not in ['gaussian', 'bernoulli', 'poisson']:
            raise ValueError("Family must be 'gaussian', 'bernoulli', or 'poisson'.")
        self.family = family
        self.lr = lr
        self.max_iter = max_iter
        self.w = None
        self.losses = []

    def fit(self, X, y):
        """Fit the model using gradient descent."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        
        y = y.reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
            
        m, n = X.shape
        # Add bias column
        X_padded = np.c_[np.ones(m), X]
        
        self.w = np.zeros(n + 1)
        self.losses = []
        
        for _ in range(self.max_iter):
            eta = X_padded @ self.w
            mu = self._inverse_link(eta)
            
            # Gradients for all three canonical link functions simplify to X.T @ (mu - y)
            grad = (1 / m) * X_padded.T @ (mu - y)
            self.w -= self.lr * grad
            
            loss = self._loss(y, mu)
            self.losses.append(loss)
            
        return self

    def predict(self, X):
        """Predict target values or class labels."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            
        X_padded = np.c_[np.ones(X.shape[0]), X]
        eta = X_padded @ self.w
        mu = self._inverse_link(eta)
        
        if self.family == 'bernoulli':
            return (mu >= 0.5).astype(int)
        return mu

    def predict_proba(self, X):
        """Predict probabilities (meaningful for Bernoulli)."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            
        X_padded = np.c_[np.ones(X.shape[0]), X]
        eta = X_padded @ self.w
        return self._inverse_link(eta)

    def _inverse_link(self, eta):
        if self.family == 'gaussian':
            return eta
        elif self.family == 'bernoulli':
            # Clip for numerical stability
            eta_clipped = np.clip(eta, -250, 250)
            return 1 / (1 + np.exp(-eta_clipped))
        elif self.family == 'poisson':
            # Prevent overflow in exp
            eta_clipped = np.clip(eta, None, 250)
            return np.exp(eta_clipped)

    def _loss(self, y, mu):
        m = y.shape[0]
        if self.family == 'gaussian':
            return (1 / (2 * m)) * np.sum((y - mu) ** 2)
        elif self.family == 'bernoulli':
            # Clip mu to avoid log(0)
            mu = np.clip(mu, 1e-15, 1 - 1e-15)
            return -(1 / m) * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
        elif self.family == 'poisson':
            # Clip mu to avoid log(0)
            mu = np.clip(mu, 1e-15, None)
            return (1 / m) * np.sum(mu - y * np.log(mu))
