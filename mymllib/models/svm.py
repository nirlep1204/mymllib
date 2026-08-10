import numpy as np
import pandas as pd

def _linear_kernel(x1, x2):
    return x1 @ x2.T

def _rbf_kernel(x1, x2, gamma):
    # ||x1-x2||^2 = ||x1||^2 + ||x2||^2 - 2*x1.x2^T
    sq1 = np.sum(x1**2, axis=1, keepdims=True)
    sq2 = np.sum(x2**2, axis=1, keepdims=True)
    return np.exp(-gamma * (sq1 + sq2.T - 2 * x1 @ x2.T))

def _poly_kernel(x1, x2, degree):
    return (1 + x1 @ x2.T) ** degree

class SVM:
    def __init__(self, C=1.0, kernel='linear', gamma=None, degree=3, tol=1e-3, max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter
        
        self.alpha = None
        self.b = 0.0
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.support_vector_alphas_ = None
        self.classes_ = None

    def _get_kernel_matrix(self, X1, X2):
        if self.kernel == 'linear':
            return _linear_kernel(X1, X2)
        elif self.kernel == 'rbf':
            gamma = self.gamma if self.gamma is not None else 1.0 / X1.shape[1]
            return _rbf_kernel(X1, X2, gamma)
        elif self.kernel == 'poly':
            return _poly_kernel(X1, X2, self.degree)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        """Fit the SVM model using SMO."""
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVM only supports binary classification.")
            
        # Map labels to -1 and 1
        y_mapped = np.where(y == self.classes_[1], 1, -1)
        
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        
        K = self._get_kernel_matrix(X, X)
        
        passes = 0
        while passes < self.max_iter:
            num_changed_alphas = 0
            for i in range(n_samples):
                # f(x_i) = sum(alpha_j * y_j * K(x_i, x_j)) + b
                f_i = np.sum(self.alpha * y_mapped * K[:, i]) + self.b
                E_i = f_i - y_mapped[i]
                
                # Check KKT conditions
                r_i = E_i * y_mapped[i]
                
                if (r_i < -self.tol and self.alpha[i] < self.C) or (r_i > self.tol and self.alpha[i] > 0):
                    # Select j != i randomly
                    j = np.random.choice([idx for idx in range(n_samples) if idx != i])
                    
                    f_j = np.sum(self.alpha * y_mapped * K[:, j]) + self.b
                    E_j = f_j - y_mapped[j]
                    
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # Compute bounds L and H
                    if y_mapped[i] != y_mapped[j]:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)
                        
                    if L == H:
                        continue
                        
                    # Compute eta
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                        
                    # Update alpha_j
                    alpha_j_new = alpha_j_old - (y_mapped[j] * (E_i - E_j)) / eta
                    
                    # Clip alpha_j
                    if alpha_j_new > H:
                        alpha_j_new = H
                    elif alpha_j_new < L:
                        alpha_j_new = L
                        
                    if abs(alpha_j_new - alpha_j_old) < self.tol:
                        continue
                        
                    # Update alpha_i
                    alpha_i_new = alpha_i_old + y_mapped[i] * y_mapped[j] * (alpha_j_old - alpha_j_new)
                    
                    # Update bias
                    b1 = self.b - E_i - y_mapped[i] * (alpha_i_new - alpha_i_old) * K[i, i] - y_mapped[j] * (alpha_j_new - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - y_mapped[i] * (alpha_i_new - alpha_i_old) * K[i, j] - y_mapped[j] * (alpha_j_new - alpha_j_old) * K[j, j]
                    
                    if 0 < alpha_i_new < self.C:
                        self.b = b1
                    elif 0 < alpha_j_new < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                        
                    self.alpha[i] = alpha_i_new
                    self.alpha[j] = alpha_j_new
                    num_changed_alphas += 1
                    
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
                
        # Store support vectors
        sv_indices = self.alpha > self.tol
        self.support_vectors_ = X[sv_indices]
        self.support_vector_labels_ = y_mapped[sv_indices]
        self.support_vector_alphas_ = self.alpha[sv_indices]
        
        return self

    def decision_function(self, X):
        """Return raw decision values (distance from hyperplane)."""
        is_pandas = False
        index = None
        if hasattr(X, 'values'):
            is_pandas = True
            index = X.index
            X = X.values
            
        if self.support_vectors_ is None:
            raise ValueError("The model is not fitted yet.")
            
        K = self._get_kernel_matrix(self.support_vectors_, X)
        
        decision = np.sum((self.support_vector_alphas_ * self.support_vector_labels_)[:, np.newaxis] * K, axis=0) + self.b
        
        if is_pandas:
            return pd.Series(decision, index=index)
        return decision

    def predict(self, X):
        """Predict class labels for samples in X."""
        decision = self.decision_function(X)
        
        if hasattr(decision, 'values'):
            decision_vals = decision.values
        else:
            decision_vals = decision
            
        y_pred_mapped = np.sign(decision_vals)
        y_pred_mapped[y_pred_mapped == 0] = 1
        
        y_pred = np.where(y_pred_mapped == 1, self.classes_[1], self.classes_[0])
        
        if hasattr(decision, 'index'):
            return pd.Series(y_pred, index=decision.index)
        return y_pred

    def support_vectors(self):
        """Return the support vectors."""
        return self.support_vectors_
