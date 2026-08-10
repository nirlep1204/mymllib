import numpy as np
import pandas as pd

class XGBoostNode:
    """Internal decision tree node for XGBoost."""
    def __init__(self, is_leaf=False, value=None, split_col=None, split_val=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.value = value
        self.split_col = split_col
        self.split_val = split_val
        self.left = left
        self.right = right

class XGBoostTree:
    """A single XGBoost regression tree built using exact greedy algorithm."""
    
    def __init__(self, max_depth=3, reg_lambda=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.root = None

    def _calc_weight(self, g, h):
        return -np.sum(g) / (np.sum(h) + self.reg_lambda)

    def _calc_gain(self, g_L, h_L, g_R, h_R):
        G_L, H_L = np.sum(g_L), np.sum(h_L)
        G_R, H_R = np.sum(g_R), np.sum(h_R)
        gain = 0.5 * (
            (G_L**2 / (H_L + self.reg_lambda)) +
            (G_R**2 / (H_R + self.reg_lambda)) -
            ((G_L + G_R)**2 / (H_L + H_R + self.reg_lambda))
        ) - self.gamma
        return gain

    def _build_tree(self, X, g, h, depth):
        if depth >= self.max_depth or len(X) < 2:
            return XGBoostNode(is_leaf=True, value=self._calc_weight(g, h))

        best_gain = 0.0
        best_split = None
        
        n_features = X.shape[1]
        for col in range(n_features):
            x_col = X[:, col]
            unique_vals = np.unique(x_col)
            
            for val in unique_vals[:-1]:
                left_mask = x_col <= val
                right_mask = ~left_mask
                
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                    
                gain = self._calc_gain(g[left_mask], h[left_mask], g[right_mask], h[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_split = (col, val, left_mask, right_mask)
                    
        if best_gain > 0 and best_split is not None:
            col, val, left_mask, right_mask = best_split
            left = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
            right = self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth + 1)
            return XGBoostNode(is_leaf=False, split_col=col, split_val=val, left=left, right=right)
            
        return XGBoostNode(is_leaf=True, value=self._calc_weight(g, h))

    def fit(self, X, g, h):
        self.root = self._build_tree(X, g, h, 0)
        return self

    def _predict_row(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.split_col] <= node.split_val:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)

    def predict(self, X):
        return np.array([self._predict_row(x, self.root) for x in X])

class XGBoost:
    """XGBoost algorithm for regression and binary classification."""
    
    def __init__(self, n_rounds=100, lr=0.1, max_depth=3, reg_lambda=1.0, gamma=0.0, task='regress'):
        self.n_rounds = n_rounds
        self.lr = lr
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.task = task
        self.trees = []
        self.initial_pred = 0.0
        self.classes = None

    def fit(self, X, y):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
        y_arr = y_arr.ravel()
        
        if self.task == 'classify':
            self.classes = np.unique(y_arr)
            y_bin = np.where(y_arr == self.classes[1], 1, 0)
            
            p = np.mean(y_bin)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            self.initial_pred = np.log(p / (1 - p))
            F = np.full(len(y_bin), self.initial_pred)
            
            for _ in range(self.n_rounds):
                p_i = 1 / (1 + np.exp(-F))
                # Logistic loss gradients
                g = p_i - y_bin
                h = p_i * (1 - p_i)
                h = np.maximum(h, 1e-10)
                
                tree = XGBoostTree(self.max_depth, self.reg_lambda, self.gamma)
                tree.fit(X_arr, g, h)
                self.trees.append(tree)
                F += self.lr * tree.predict(X_arr)
                
        else:
            self.initial_pred = np.mean(y_arr)
            F = np.full(len(y_arr), self.initial_pred)
            
            for _ in range(self.n_rounds):
                # MSE gradients
                g = F - y_arr
                h = np.ones_like(y_arr)
                
                tree = XGBoostTree(self.max_depth, self.reg_lambda, self.gamma)
                tree.fit(X_arr, g, h)
                self.trees.append(tree)
                F += self.lr * tree.predict(X_arr)
                
        return self

    def predict_proba(self, X):
        if self.task != 'classify':
            raise ValueError("predict_proba is only for classification")
            
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        F = np.full(len(X_arr), self.initial_pred)
        for tree in self.trees:
            F += self.lr * tree.predict(X_arr)
            
        p = 1 / (1 + np.exp(-F))
        return p

    def predict(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if self.task == 'classify':
            p = self.predict_proba(X)
            preds = np.where(p >= 0.5, self.classes[1], self.classes[0])
            if isinstance(X, pd.DataFrame):
                return pd.Series(preds, index=X.index)
            return preds
        else:
            F = np.full(len(X_arr), self.initial_pred)
            for tree in self.trees:
                F += self.lr * tree.predict(X_arr)
            if isinstance(X, pd.DataFrame):
                return pd.Series(F, index=X.index)
            return F
