import numpy as np
from typing import Optional

class Dense:
    """Fully connected layer."""
    
    def __init__(self, in_size: int, out_size: int, activation: Optional[str] = 'relu') -> None:
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        
        if activation == 'relu':
            # He initialization
            self.W = np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size)
        else:
            # Xavier initialization
            self.W = np.random.randn(in_size, out_size) * np.sqrt(2.0 / (in_size + out_size))
            
        self.b = np.zeros((1, out_size))
        
        # Caches
        self.X: Optional[np.ndarray] = None
        self.Z: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None
        
        # Optimizer caches
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)
        self.s_W = np.zeros_like(self.W)
        self.s_b = np.zeros_like(self.b)
        
    def _act(self, Z: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'sigmoid':
            # clip for numerical stability
            Z = np.clip(Z, -500, 500)
            return 1.0 / (1.0 + np.exp(-Z))
        elif self.activation == 'tanh':
            return np.tanh(Z)
        elif self.activation == 'softmax':
            # shift for numerical stability
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        elif self.activation in ('linear', None):
            return Z
        raise ValueError(f"Unknown activation: {self.activation}")
        
    def _act_deriv(self, dA: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return dA * (self.Z > 0) # type: ignore
        elif self.activation == 'sigmoid':
            sig = self.A
            return dA * sig * (1 - sig) # type: ignore
        elif self.activation == 'tanh':
            return dA * (1 - self.A ** 2) # type: ignore
        elif self.activation == 'softmax':
            # Assume dA from cross-entropy loss handles the derivative (dA = A - Y)
            return dA
        elif self.activation in ('linear', None):
            return dA
        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.X = X
        self.Z = np.dot(X, self.W) + self.b
        self.A = self._act(self.Z)
        return self.A
        
    def backward(self, dA: np.ndarray) -> np.ndarray:
        """Backward pass."""
        m = self.X.shape[0] # type: ignore
        
        dZ = self._act_deriv(dA)
        
        self.dW = np.dot(self.X.T, dZ) / m # type: ignore
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = np.dot(dZ, self.W.T)
        
        return dX
