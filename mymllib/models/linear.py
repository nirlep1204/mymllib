import numpy as np


class Linear:
    def __init__(self, method="gd", lr=0.01, max_iter=1000, batch_size=None, reg=None, alpha=0.0, seed=None):
        """Initialize Linear Regression.
        
        Parameters:
        - method: 'gd' (gradient descent) or 'normal'/'closed' (normal equation)
        - lr: learning rate for gradient descent
        - max_iter: number of iterations/epochs
        - batch_size: None for Batch GD, 1 for Stochastic GD (SGD), or int (e.g. 32) for Mini-Batch GD
        - reg: 'l1', 'l2', or None
        - alpha: regularization strength
        - seed: random seed for shuffling in SGD / Mini-Batch GD
        """
        self.method = str(method).strip("'\" \t\r\n").lower() if method is not None else "gd"
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.reg = str(reg).strip("'\" \t\r\n").lower() if reg is not None else None
        self.alpha = float(alpha)
        self.seed = int(seed) if seed is not None else None
        self.weights = None
        self.losses = []

    @property
    def w(self):
        """Feature weights / coefficients (excluding intercept)."""
        if self.weights is None:
            return None
        return self.weights[1:]

    @property
    def b(self):
        """Bias / intercept term."""
        if self.weights is None:
            return None
        return float(self.weights[0])

    def fit(self, X, y):
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

        if self.method in ("normal", "closed", "exact"):
            if self.reg == "l2":
                I = np.eye(n + 1)
                I[0, 0] = 0
                self.weights = np.linalg.pinv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
            else:
                self.weights = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            
            # Compute final single-step cost for reference
            full_preds = X_b @ self.weights
            final_cost = (1 / (2 * m)) * np.sum((full_preds - y) ** 2)
            self.losses = [final_cost]
        elif self.method in ("gd", "gradient_descent"):
            self.weights = np.zeros(n + 1)
            
            # Record initial baseline loss before gradient steps
            init_preds = X_b @ self.weights
            init_cost = (1 / (2 * m)) * np.sum((init_preds - y) ** 2)
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
                    
                    preds = X_batch @ self.weights
                    error = preds - y_batch
                    grad = (1 / k) * (X_batch.T @ error)
                    
                    if self.reg == "l2":
                        reg_grad = (self.alpha / k) * self.weights
                        reg_grad[0] = 0
                        grad += reg_grad
                    elif self.reg == "l1":
                        reg_grad = (self.alpha / k) * np.sign(self.weights)
                        reg_grad[0] = 0
                        grad += reg_grad
                    
                    self.weights -= self.lr * grad
                
                full_preds = X_b @ self.weights
                cost = (1 / (2 * m)) * np.sum((full_preds - y) ** 2)
                self.losses.append(cost)
        else:
            raise ValueError("Method must be 'gd' or 'normal'.")

        return self

    def predict(self, X):
        """Predict target values."""
        if self.weights is None:
            raise ValueError("Model has not been fitted yet. Call .fit() first.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.weights

    def cost(self, X, y):
        """Calculate MSE cost."""
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim > 1:
            y = y.squeeze()
        
        preds = self.predict(X)
        m = len(y)
        return (1 / (2 * m)) * np.sum((preds - y) ** 2)
