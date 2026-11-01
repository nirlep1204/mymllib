import numpy as np
import pandas as pd
from typing import List, Optional, Any, Union
from .optimizers import Adam
from mymllib.autograd import Tensor

class Network:
    """Neural Network."""
    
    def __init__(self, layers: List[Any]) -> None:
        self.layers = layers
        self.losses: List[float] = []
        self.classes: Optional[np.ndarray] = None
        self.is_classifier = False
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame, pd.Series], optimizer: Optional[Any] = None, loss: str = 'cross_entropy', epochs: int = 100, batch_size: int = 32) -> 'Network':
        """Train the neural network using Tensor autograd."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values # type: ignore
            
        if optimizer is None:
            optimizer = Adam(lr=0.001)
            
        n_samples = X.shape[0]
        
        if loss == 'cross_entropy':
            self.is_classifier = True
            # One-hot encoding
            self.classes = np.unique(y)
            n_classes = len(self.classes)
            
            y_encoded = np.zeros((n_samples, n_classes))
            for i, c in enumerate(self.classes):
                y_encoded[y == c, i] = 1
            y_train = y_encoded
        else:
            self.is_classifier = False
            y_arr = np.asarray(y)
            y_train = y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr
            
        self.losses = []
        t = 0
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = Tensor(X_shuffled[i:i+batch_size])
                y_batch = Tensor(y_shuffled[i:i+batch_size])
                
                # Zero gradients for all layers
                for layer in self.layers:
                    layer.W.zero_grad()
                    layer.b.zero_grad()
                
                # Forward
                A = X_batch
                for layer in self.layers:
                    A = layer.forward(A)
                    
                # Loss computation (building the graph)
                if loss == 'cross_entropy':
                    # Add epsilon for numerical stability, already handled in Tensor.log()
                    batch_loss = (y_batch * A.log()).sum(axis=1).mean() * -1
                elif loss == 'mse':
                    batch_loss = ((A - y_batch) ** 2).mean()
                else:
                    raise ValueError(f"Unknown loss: {loss}")
                    
                epoch_loss += float(batch_loss.data)
                n_batches += 1
                
                # Backward pass automatically computes all gradients!
                batch_loss.backward()
                    
                # Update
                t += 1
                for layer in self.layers:
                    if hasattr(optimizer, 'beta1'): # Adam uses t
                        optimizer.update(layer, t)
                    else:
                        optimizer.update(layer) # type: ignore
                        
            self.losses.append(float(epoch_loss / n_batches))
            
        return self
        
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities for classification."""
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        A = Tensor(X)
        for layer in self.layers:
            A = layer.forward(A)
        return A.data
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict classes or continuous values."""
        preds = self.predict_proba(X)
        
        if self.is_classifier:
            if preds.shape[1] == 1:
                idx = (preds >= 0.5).astype(int).ravel()
                return self.classes[np.clip(idx, 0, len(self.classes)-1)] # type: ignore
            else:
                idx = np.argmax(preds, axis=1)
                return self.classes[np.clip(idx, 0, len(self.classes)-1)] # type: ignore
        else:
            return preds.ravel() if preds.shape[1] == 1 else preds
