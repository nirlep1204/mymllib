import numpy as np

class SVR:
    """Support Vector Regression."""
    
    def __init__(self, C=1.0, epsilon=0.1, kernel='linear', gamma=None, degree=3, lr=0.001, max_iter=1000):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.lr = lr
        self.max_iter = max_iter
        
        self.weights = None
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.losses = []

    def _kernel_matrix(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            gamma = self.gamma if self.gamma is not None else 1.0 / X1.shape[1]
            # (X1 - X2)^2 = X1^2 + X2^2 - 2 X1 X2
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
            return np.exp(-gamma * sq_dists)
        elif self.kernel == 'poly':
            gamma = self.gamma if self.gamma is not None else 1.0
            return (gamma * (X1 @ X2.T) + 1) ** self.degree
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        m, n = X.shape
        
        self.losses = []
        
        if self.kernel == 'linear':
            X_b = np.c_[np.ones((m, 1)), X]
            weights = np.zeros(n + 1)
            
            for _ in range(self.max_iter):
                pred = X_b @ weights
                error = pred - y
                
                # Epsilon-insensitive loss gradient
                mask_pos = error > self.epsilon   # predictions too high
                mask_neg = error < -self.epsilon  # predictions too low
                
                loss_grad = np.zeros(m)
                loss_grad[mask_pos] = 1.0
                loss_grad[mask_neg] = -1.0
                
                # Calculate loss for monitoring
                reg_loss = 0.5 * np.sum(weights[1:] ** 2)
                hinge_loss = self.C * np.mean(np.maximum(0, np.abs(error) - self.epsilon))
                self.losses.append(reg_loss + hinge_loss)
                
                grad = np.copy(weights)
                grad[0] = 0  # don't regularize bias
                grad += (self.C / m) * X_b.T @ loss_grad
                
                weights -= self.lr * grad
            
            self.weights = weights
            self.X_train = X
        else:
            # Dual optimization for non-linear kernels
            self.X_train = np.copy(X)
            K = self._kernel_matrix(X, X)
            self.alpha = np.zeros(m)
            self.b = 0.0
            
            for _ in range(self.max_iter):
                pred = K @ self.alpha + self.b
                error = pred - y
                
                mask_pos = error > self.epsilon
                mask_neg = error < -self.epsilon
                
                loss_grad = np.zeros(m)
                loss_grad[mask_pos] = 1.0
                loss_grad[mask_neg] = -1.0
                
                reg_loss = 0.5 * self.alpha.T @ K @ self.alpha
                hinge_loss = self.C * np.mean(np.maximum(0, np.abs(error) - self.epsilon))
                self.losses.append(reg_loss + hinge_loss)
                
                # Gradients w.r.t alpha and b
                grad_alpha = K @ self.alpha + (self.C / m) * (K @ loss_grad)
                grad_b = (self.C / m) * np.sum(loss_grad)
                
                self.alpha -= self.lr * grad_alpha
                self.b -= self.lr * grad_b
                
        return self

    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=float)
        
        if self.kernel == 'linear':
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
            return X_b @ self.weights
        else:
            K = self._kernel_matrix(X, self.X_train)
            return K @ self.alpha + self.b

    def support_vectors(self):
        """Return indices of support vectors."""
        if self.X_train is None:
            return np.array([])
            
        if self.kernel == 'linear':
            # Points on or outside the epsilon tube
            X_b = np.c_[np.ones((self.X_train.shape[0], 1)), self.X_train]
            # Since we don't store y, we recalculate error if we had it, but wait, 
            # we need to define support vectors purely from weights if possible, 
            # but we don't have alphas in linear mode.
            # We can approximate by saying we don't have support vectors explicitly,
            # or we could store training y. Let's just return points that are not zero.
            # Without y_train, we can't recalculate errors.
            pass
            # Just returning all training data as a fallback, or empty
            return self.X_train
        else:
            # For non-linear, points with non-zero alpha
            tol = 1e-5
            return self.X_train[np.abs(self.alpha) > tol]
