import numpy as np
import pandas as pd
from .tree import Tree

class GradientBoost:
    """Gradient Boosting for regression and binary classification."""
    
    def __init__(self, n_rounds=100, lr=0.1, max_depth=3, task='regress'):
        self.n_rounds = n_rounds
        self.lr = lr
        self.max_depth = max_depth
        self.task = task
        self.estimators = []
        self.initial_pred = None
        self.classes = None

    def fit(self, X, y):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
        y_arr = y_arr.ravel()
        
        if self.task == 'classify':
            self.classes = np.unique(y_arr)
            if len(self.classes) != 2:
                raise ValueError("GradientBoost currently supports binary classification only.")
                
            y_bin = np.where(y_arr == self.classes[1], 1, 0)
            p = np.mean(y_bin)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            self.initial_pred = np.log(p / (1 - p))
            F = np.full(len(y_bin), self.initial_pred)
            
            for _ in range(self.n_rounds):
                p_i = 1 / (1 + np.exp(-F))
                r = y_bin - p_i
                
                tree = Tree(max_depth=self.max_depth, task='regress')
                if isinstance(X, pd.DataFrame):
                    tree.fit(X, pd.Series(r))
                else:
                    tree.fit(X_arr, r)
                    
                self.estimators.append(tree)
                
                pred_r = tree.predict(X)
                if isinstance(pred_r, pd.Series):
                    pred_r = pred_r.values
                F = F + self.lr * pred_r
                
        else:
            self.initial_pred = np.mean(y_arr)
            F = np.full(len(y_arr), self.initial_pred)
            
            for _ in range(self.n_rounds):
                r = y_arr - F
                tree = Tree(max_depth=self.max_depth, task='regress')
                
                if isinstance(X, pd.DataFrame):
                    tree.fit(X, pd.Series(r))
                else:
                    tree.fit(X_arr, r)
                    
                self.estimators.append(tree)
                
                pred_r = tree.predict(X)
                if isinstance(pred_r, pd.Series):
                    pred_r = pred_r.values
                F = F + self.lr * pred_r
                
        return self

    def predict_proba(self, X):
        if self.task != 'classify':
            raise ValueError("predict_proba is only for classification")
            
        F = np.full(len(X), self.initial_pred)
        for tree in self.estimators:
            pred_r = tree.predict(X)
            if isinstance(pred_r, pd.Series):
                pred_r = pred_r.values
            F = F + self.lr * pred_r
            
        p = 1 / (1 + np.exp(-F))
        return p

    def predict(self, X):
        if self.task == 'classify':
            p = self.predict_proba(X)
            preds = np.where(p >= 0.5, self.classes[1], self.classes[0])
            if isinstance(X, pd.DataFrame):
                return pd.Series(preds, index=X.index)
            return preds
        else:
            F = np.full(len(X), self.initial_pred)
            for tree in self.estimators:
                pred_r = tree.predict(X)
                if isinstance(pred_r, pd.Series):
                    pred_r = pred_r.values
                F = F + self.lr * pred_r
            if isinstance(X, pd.DataFrame):
                return pd.Series(F, index=X.index)
            return F
