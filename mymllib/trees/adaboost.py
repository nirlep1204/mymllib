import numpy as np
import pandas as pd
from .tree import Tree

class AdaBoost:
    """AdaBoost classifier."""
    
    def __init__(self, n_rounds=50, lr=1.0):
        self.n_rounds = n_rounds
        self.lr = lr
        self.estimators = []
        self.estimator_weights = []
        self.classes = None

    def fit(self, X, y):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
        y_arr = y_arr.ravel()
        
        self.classes = np.unique(y_arr)
        K = len(self.classes)
        m = len(X_arr)
        w = np.ones(m) / m
        
        for _ in range(self.n_rounds):
            # Bootstrap sample according to weights
            indices = np.random.choice(m, size=m, p=w, replace=True)
            X_sample = X_arr[indices]
            y_sample = y_arr[indices]
            
            stump = Tree(max_depth=1, task='classify')
            if isinstance(X, pd.DataFrame):
                stump.fit(X.iloc[indices], pd.Series(y_sample))
            else:
                stump.fit(X_sample, y_sample)
                
            pred = stump.predict(X_arr)
            if isinstance(pred, pd.Series):
                pred = pred.values
                
            incorrect = pred != y_arr
            err = np.sum(w[incorrect]) / np.sum(w)
            
            if err >= 1 - 1/K or err == 0:
                if err == 0:
                    alpha = self.lr * (np.log((1 - err) / (err + 1e-10)) + np.log(K - 1))
                    self.estimators.append(stump)
                    self.estimator_weights.append(alpha)
                break
                
            # SAMME alpha
            alpha = self.lr * (np.log((1 - err) / (err + 1e-10)) + np.log(K - 1))
            
            w = w * np.exp(alpha * incorrect)
            w = w / np.sum(w)
            
            self.estimators.append(stump)
            self.estimator_weights.append(alpha)
            
        return self

    def predict(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        n = len(X_arr)
        class_scores = {c: np.zeros(n) for c in self.classes}
        
        for stump, alpha in zip(self.estimators, self.estimator_weights):
            pred = stump.predict(X)
            if isinstance(pred, pd.Series):
                pred = pred.values
            for i in range(n):
                class_scores[pred[i]][i] += alpha
                
        scores = np.zeros((n, len(self.classes)))
        for i, c in enumerate(self.classes):
            scores[:, i] = class_scores[c]
            
        pred_indices = np.argmax(scores, axis=1)
        preds = self.classes[pred_indices]
        
        if isinstance(X, pd.DataFrame):
            return pd.Series(preds, index=X.index)
        return preds
