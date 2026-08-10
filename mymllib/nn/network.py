import numpy as np
import pandas as pd
from .optimizers import Adam

class Network:
    """Neural Network."""
    
    def __init__(self, layers):
        self.layers = layers
        self.losses = []
        self.classes = None
        self.is_classifier = False
        
    def fit(self, X, y, optimizer=None, loss='cross_entropy', epochs=100, batch_size=32):
        """Train the neural network."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
            
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
            y_train = y.reshape(-1, 1) if y.ndim == 1 else y
            
        self.losses = []
        t = 0
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward
                A = X_batch
                for layer in self.layers:
                    A = layer.forward(A)
                    
                # Loss & dA
                if loss == 'cross_entropy':
                    # Add epsilon for numerical stability
                    eps = 1e-15
                    A_clipped = np.clip(A, eps, 1 - eps)
                    batch_loss = -np.mean(np.sum(y_batch * np.log(A_clipped), axis=1))
                    
                    # Derivative of Cross-Entropy with Softmax is A - y
                    dA = A - y_batch
                elif loss == 'mse':
                    batch_loss = np.mean((A - y_batch) ** 2)
                    dA = 2 * (A - y_batch) / A.shape[1]
                else:
                    raise ValueError(f"Unknown loss: {loss}")
                    
                epoch_loss += batch_loss
                n_batches += 1
                
                # Backward
                for layer in reversed(self.layers):
                    dA = layer.backward(dA)
                    
                # Update
                t += 1
                for layer in self.layers:
                    if hasattr(optimizer, 'beta1'): # Adam uses t
                        optimizer.update(layer, t)
                    else:
                        optimizer.update(layer)
                        
            self.losses.append(epoch_loss / n_batches)
            
        return self
        
    def predict_proba(self, X):
        """Predict probabilities for classification."""
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
        
    def predict(self, X):
        """Predict classes or continuous values."""
        preds = self.predict_proba(X)
        
        if self.is_classifier:
            if preds.shape[1] == 1:
                idx = (preds >= 0.5).astype(int).ravel()
                return self.classes[idx]
            else:
                idx = np.argmax(preds, axis=1)
                return self.classes[idx]
        else:
            return preds.ravel() if preds.shape[1] == 1 else preds
