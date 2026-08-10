import numpy as np
import pandas as pd
from typing import Union, Any

class Perceptron:
    """Classical single-layer Rosenblatt Perceptron for binary classification."""
    
    def __init__(self, lr: float = 0.01, max_iter: int = 1000) -> None:
        self.lr = lr
        self.max_iter = max_iter
        self.w: Any = None
        self.b: Any = None
        self.errors: List[Any] = [] # type: ignore
        self.classes: Any = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame, pd.Series]) -> "Perceptron":
        """Fit the model to the training data."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values # type: ignore
            
        y = y.ravel() # type: ignore
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
            
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError("Perceptron is for binary classification only.")
            
        y_internal = np.where(y == self.classes[1], 1, -1)
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.errors = []
        
        for _ in range(self.max_iter):
            errors_epoch = 0
            for i in range(n_samples):
                xi = X[i]
                yi = y_internal[i]
                y_hat = np.sign(np.dot(self.w, xi) + self.b)
                
                # sign(0) is 0, treat as -1
                if y_hat == 0:
                    y_hat = -1
                    
                if y_hat != yi:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors_epoch += 1
            
            self.errors.append(errors_epoch)
            if errors_epoch == 0:
                break
                
        return self
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict classes for samples in X."""
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        z = np.dot(X, self.w) + self.b
        y_pred = np.where(z > 0.0, 1, -1)
        
        return np.where(y_pred == 1, self.classes[1], self.classes[0])
