import numpy as np
from typing import Optional, Any

class SGD:
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
        
    def update(self, layer: Any, t: Optional[int] = None) -> None:
        layer.W.data -= self.lr * layer.W.grad
        layer.b.data -= self.lr * layer.b.grad

class Momentum:
    """Momentum optimizer."""
    
    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        self.lr = lr
        self.beta = beta
        
    def update(self, layer: Any, t: Optional[int] = None) -> None:
        layer.v_W = self.beta * layer.v_W + (1 - self.beta) * layer.W.grad
        layer.v_b = self.beta * layer.v_b + (1 - self.beta) * layer.b.grad
        
        layer.W.data -= self.lr * layer.v_W
        layer.b.data -= self.lr * layer.v_b

class RMSProp:
    """RMSProp optimizer."""
    
    def __init__(self, lr: float = 0.001, beta: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta = beta
        self.eps = eps
        
    def update(self, layer: Any, t: Optional[int] = None) -> None:
        layer.s_W = self.beta * layer.s_W + (1 - self.beta) * (layer.W.grad ** 2)
        layer.s_b = self.beta * layer.s_b + (1 - self.beta) * (layer.b.grad ** 2)
        
        layer.W.data -= self.lr * layer.W.grad / (np.sqrt(layer.s_W) + self.eps)
        layer.b.data -= self.lr * layer.b.grad / (np.sqrt(layer.s_b) + self.eps)

class Adam:
    """Adam optimizer."""
    
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
    def update(self, layer: Any, t: int) -> None:
        layer.v_W = self.beta1 * layer.v_W + (1 - self.beta1) * layer.W.grad
        layer.v_b = self.beta1 * layer.v_b + (1 - self.beta1) * layer.b.grad
        
        layer.s_W = self.beta2 * layer.s_W + (1 - self.beta2) * (layer.W.grad ** 2)
        layer.s_b = self.beta2 * layer.s_b + (1 - self.beta2) * (layer.b.grad ** 2)
        
        # Bias correction
        v_W_corr = layer.v_W / (1 - self.beta1 ** t)
        v_b_corr = layer.v_b / (1 - self.beta1 ** t)
        
        s_W_corr = layer.s_W / (1 - self.beta2 ** t)
        s_b_corr = layer.s_b / (1 - self.beta2 ** t)
        
        layer.W.data -= self.lr * v_W_corr / (np.sqrt(s_W_corr) + self.eps)
        layer.b.data -= self.lr * v_b_corr / (np.sqrt(s_b_corr) + self.eps)
