import numpy as np
from typing import Union, Optional, Any

class Logistic:
    def __init__(self, lr: float = 0.01, max_iter: int = 1000, batch_size: Optional[int] = None, reg: Optional[str] = None, alpha: float = 0.0, seed: Optional[int] = None) -> None:
        """Initialize Logistic Regression.
        
        Parameters:
        - lr: learning rate
        - max_iter: number of iterations/epochs
        - batch_size: None for Batch GD, 1 for Stochastic GD (SGD), or int (e.g. 32) for Mini-Batch GD
        - reg: 'l1', 'l2', or None
        - alpha: regularization strength
        - seed: random seed for shuffling in SGD / Mini-Batch GD
        """
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.reg = str(reg).strip("'\" \t\r\n").lower() if reg is not None else None
        self.alpha = float(alpha)
        self.seed = int(seed) if seed is not None else None
        self.weights = None
        self.losses = [] # type: ignore

    @property
    def w(self) -> Optional[np.ndarray]:
        if self.weights is None:
            return None
        return self.weights[1:]

    @property
    def b(self) -> Optional[float]:
        if self.weights is None:
            return None
        return float(self.weights[0])

    def fit(self, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any]) -> "Logistic":
        """Fit the model to the data."""
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim > 1:
            y = y.squeeze()
            
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        self.weights = np.zeros(n + 1) # type: ignore
        
        # Initial baseline loss
        full_z = np.clip(X_b @ self.weights, -500, 500)
        full_h = 1 / (1 + np.exp(-full_z))
        h_safe = np.clip(full_h, 1e-15, 1 - 1e-15)
        init_cost = -(1 / m) * np.sum(y * np.log(h_safe) + (1 - y) * np.log(1 - h_safe))
        self.losses = [init_cost]
        
        rng = np.random.default_rng(self.seed)
        b_size = m if (self.batch_size is None or self.batch_size >= m) else max(1, int(self.batch_size))

        for _ in range(self.max_iter):
            if b_size < m:
                indices = rng.permutation(m)
                X_b_shuffled = X_b[indices]
                y_shuffled = y[indices]
            else:
                X_b_shuffled = X_b
                y_shuffled = y

            for start in range(0, m, b_size):
                end = min(start + b_size, m)
                X_batch = X_b_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                k = end - start

                z = X_batch @ self.weights
                z = np.clip(z, -500, 500)
                h = 1 / (1 + np.exp(-z))
                
                error = h - y_batch
                grad = (1 / k) * (X_batch.T @ error)
                
                if self.reg == "l2":
                    reg_grad = (self.alpha / k) * self.weights
                    reg_grad[0] = 0
                    grad += reg_grad
                elif self.reg == "l1":
                    reg_grad = (self.alpha / k) * np.sign(self.weights) # type: ignore
                    reg_grad[0] = 0
                    grad += reg_grad
                    
                self.weights -= self.lr * grad
            
            # Record full-epoch cost
            full_z = np.clip(X_b @ self.weights, -500, 500)
            full_h = 1 / (1 + np.exp(-full_z))
            h_safe = np.clip(full_h, 1e-15, 1 - 1e-15)
            cost = -(1 / m) * np.sum(y * np.log(h_safe) + (1 - y) * np.log(1 - h_safe))
            self.losses.append(cost)

        return self

    def predict_proba(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Predict probability of class 1."""
        if self.weights is None:
            raise ValueError("Model has not been fitted yet. Call .fit() first.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        z = X_b @ self.weights
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def cost(self, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any]) -> float:
        """Calculate cross-entropy cost."""
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim > 1:
            y = y.squeeze()
            
        m = len(y)
        h = self.predict_proba(X)
        h_safe = np.clip(h, 1e-15, 1 - 1e-15)
        return -(1 / m) * np.sum(y * np.log(h_safe) + (1 - y) * np.log(1 - h_safe))
