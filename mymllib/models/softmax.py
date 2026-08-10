import numpy as np

def _softmax(z):
    """Compute softmax values for each sets of scores in z."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class Softmax:
    def __init__(self, lr=0.01, max_iter=1000, batch_size=None, seed=None):
        """Initialize the Softmax model.
        
        Parameters:
        - lr: learning rate
        - max_iter: number of iterations/epochs
        - batch_size: None for Batch GD, 1 for Stochastic GD (SGD), or int (e.g. 32) for Mini-Batch GD
        - seed: random seed for shuffling in SGD / Mini-Batch GD
        """
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.seed = seed
        self.weights = None
        self.classes = None
        self.losses = []

    def fit(self, X, y):
        """Fit the model to the training data."""
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        
        self.classes = np.unique(y)
        k = len(self.classes)
        
        # one-hot encode y
        Y = np.zeros((m, k))
        for i, c in enumerate(self.classes):
            Y[:, i] = (y == c).astype(float)
            
        self.weights = np.zeros((n + 1, k))
        self.losses = []
        rng = np.random.default_rng(self.seed)

        b_size = m if (self.batch_size is None or self.batch_size >= m) else max(1, int(self.batch_size))

        for _ in range(self.max_iter):
            if b_size < m:
                indices = rng.permutation(m)
                X_b_shuffled = X_b[indices]
                Y_shuffled = Y[indices]
            else:
                X_b_shuffled = X_b
                Y_shuffled = Y

            for start in range(0, m, b_size):
                end = min(start + b_size, m)
                X_batch = X_b_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                batch_k = end - start

                logits = X_batch @ self.weights
                probs = _softmax(logits)
                
                grad = (1 / batch_k) * X_batch.T @ (probs - Y_batch)
                self.weights -= self.lr * grad
            
            # Record full-epoch loss
            full_logits = X_b @ self.weights
            full_probs = _softmax(full_logits)
            probs_clipped = np.clip(full_probs, 1e-15, 1.0)
            loss = -(1 / m) * np.sum(Y * np.log(probs_clipped))
            self.losses.append(loss)
            
        return self

    def predict_proba(self, X):
        """Return probability estimates for each class."""
        if hasattr(X, 'values'):
            X = X.values
            
        X = np.asarray(X, dtype=float)
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        
        logits = X_b @ self.weights
        return _softmax(logits)

    def predict(self, X):
        """Predict class labels for samples in X."""
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes[class_indices]
