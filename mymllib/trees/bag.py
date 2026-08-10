import numpy as np
import pandas as pd
import copy
from .tree import Tree

class Bag:
    def __init__(self, base_model=None, n_models=10, seed=None):
        self.base_model = base_model
        self.n_models = n_models
        self.seed = seed
        self.models = []
        
    def fit(self, X, y):
        """Fit bagging ensemble."""
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_arr = X.values
        else:
            X_arr = X
            
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = y

        if self.base_model is None:
            # infer task: if continuous float or many unique, regress
            if np.issubdtype(y_arr.dtype, np.floating) or (np.issubdtype(y_arr.dtype, np.number) and len(np.unique(y_arr)) > 20):
                self.base_model = Tree(task='regress')
            else:
                self.base_model = Tree(task='classify')
                
        self.models = []
        rng = np.random.default_rng(self.seed)
        
        n_samples = X_arr.shape[0]
        
        for _ in range(self.n_models):
            model = copy.deepcopy(self.base_model)
            
            # allow model to have variance if it has a seed
            if hasattr(model, 'seed'):
                model.seed = int(rng.integers(0, 1000000))
            elif hasattr(model, 'random_state'):
                model.random_state = int(rng.integers(0, 1000000))
                
            # bootstrap sampling
            idxs = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_arr[idxs]
            y_boot = y_arr[idxs]
            
            model.fit(X_boot, y_boot)
            self.models.append(model)
            
        return self
        
    def predict(self, X):
        """Aggregate predictions from models."""
        preds = np.array([model.predict(X) for model in self.models])
        
        # Check if classification or regression
        is_numeric = np.issubdtype(preds.dtype, np.number)
        is_float = np.issubdtype(preds.dtype, np.floating)
        
        if not is_numeric or (is_float and not np.all(preds == np.floor(preds))):
            # regression
            return np.mean(preds, axis=0)
        else:
            # classification majority vote
            def majority_vote(arr):
                unique, counts = np.unique(arr, return_counts=True)
                return unique[np.argmax(counts)]
                
            return np.apply_along_axis(majority_vote, 0, preds)
            
    def predict_proba(self, X):
        """Average probabilities if supported."""
        if not hasattr(self.models[0], 'predict_proba'):
            raise ValueError("Base model does not support predict_proba.")
            
        all_probas = np.array([model.predict_proba(X) for model in self.models])
        return np.mean(all_probas, axis=0)
