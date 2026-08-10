import numpy as np

class SGD:
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, layer, t=None):
        layer.W -= self.lr * layer.dW
        layer.b -= self.lr * layer.db

class Momentum:
    """Momentum optimizer."""
    
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        
    def update(self, layer, t=None):
        layer.v_W = self.beta * layer.v_W + (1 - self.beta) * layer.dW
        layer.v_b = self.beta * layer.v_b + (1 - self.beta) * layer.db
        
        layer.W -= self.lr * layer.v_W
        layer.b -= self.lr * layer.v_b

class RMSProp:
    """RMSProp optimizer."""
    
    def __init__(self, lr=0.001, beta=0.999, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        
    def update(self, layer, t=None):
        layer.s_W = self.beta * layer.s_W + (1 - self.beta) * (layer.dW ** 2)
        layer.s_b = self.beta * layer.s_b + (1 - self.beta) * (layer.db ** 2)
        
        layer.W -= self.lr * layer.dW / (np.sqrt(layer.s_W) + self.eps)
        layer.b -= self.lr * layer.db / (np.sqrt(layer.s_b) + self.eps)

class Adam:
    """Adam optimizer."""
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
    def update(self, layer, t):
        layer.v_W = self.beta1 * layer.v_W + (1 - self.beta1) * layer.dW
        layer.v_b = self.beta1 * layer.v_b + (1 - self.beta1) * layer.db
        
        layer.s_W = self.beta2 * layer.s_W + (1 - self.beta2) * (layer.dW ** 2)
        layer.s_b = self.beta2 * layer.s_b + (1 - self.beta2) * (layer.db ** 2)
        
        # Bias correction
        v_W_corr = layer.v_W / (1 - self.beta1 ** t)
        v_b_corr = layer.v_b / (1 - self.beta1 ** t)
        
        s_W_corr = layer.s_W / (1 - self.beta2 ** t)
        s_b_corr = layer.s_b / (1 - self.beta2 ** t)
        
        layer.W -= self.lr * v_W_corr / (np.sqrt(s_W_corr) + self.eps)
        layer.b -= self.lr * v_b_corr / (np.sqrt(s_b_corr) + self.eps)
