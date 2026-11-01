import numpy as np
from typing import Optional, Union
from mymllib.autograd import Tensor

class Dense:
    """Fully connected layer using Tensor autograd engine."""
    
    def __init__(self, in_size: int, out_size: int, activation: Optional[str] = 'relu') -> None:
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        
        if activation == 'relu':
            # He initialization
            W_data = np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size)
        else:
            # Xavier initialization
            W_data = np.random.randn(in_size, out_size) * np.sqrt(2.0 / (in_size + out_size))
            
        b_data = np.zeros((1, out_size))
        
        self.W = Tensor(W_data)
        self.b = Tensor(b_data)
        
        # Optimizer caches
        self.v_W = np.zeros_like(W_data)
        self.v_b = np.zeros_like(b_data)
        self.s_W = np.zeros_like(W_data)
        self.s_b = np.zeros_like(b_data)
        
    def _act(self, Z: Tensor) -> Tensor:
        if self.activation == 'relu':
            return Z.relu()
        elif self.activation == 'sigmoid':
            return Z.sigmoid()
        elif self.activation == 'tanh':
            return Z.tanh()
        elif self.activation == 'softmax':
            # shift for numerical stability
            # Since Tensor doesn't have max across axis natively without some work, we can just do standard Softmax
            # We implemented max(axis=1) in Tensor!
            m = Z.max(axis=1, keepdims=True)
            exp_Z = (Z - m).exp()
            return exp_Z / exp_Z.sum(axis=1, keepdims=True)
        elif self.activation in ('linear', None):
            return Z
        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, X: Union[np.ndarray, Tensor]) -> Tensor:
        """Forward pass."""
        if not isinstance(X, Tensor):
            X = Tensor(X)
        self.Z = X @ self.W + self.b
        self.A = self._act(self.Z)
        return self.A
