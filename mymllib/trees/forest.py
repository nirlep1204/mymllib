import numpy as np
import pandas as pd
from typing import Optional, Any, Union, List
from .tree import Tree


class Forest:
    def __init__(self, n_trees: int = 100, task: str = "classify", max_depth: int = 10, min_samples_split: int = 2, max_features: Union[str, float, int] = "sqrt", seed: Optional[int] = None) -> None:
        self.n_trees = int(n_trees)
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.trees: List[Any] = []
        self.classes: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: Union[pd.DataFrame, pd.Series, np.ndarray], y: Union[pd.DataFrame, pd.Series, np.ndarray]) -> "Forest":
        """Fit random forest ensemble."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_arr = X.values
        else:
            X_arr = X

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_arr = y.values
        else:
            y_arr = y

        X_arr = np.asarray(X_arr)
        y_arr = np.asarray(y_arr)

        if self.task == "classify":
            self.classes = np.unique(y_arr)

        self.trees = []
        rng = np.random.default_rng(self.seed)

        n_samples = X_arr.shape[0]

        for _ in range(self.n_trees):
            tree_seed = int(rng.integers(0, 1000000))
            tree = Tree(
                task=self.task,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                seed=tree_seed,
            )

            idxs = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_arr[idxs]
            y_boot = y_arr[idxs]

            if self.task == "classify":
                tree.fit(X_boot, y_boot, classes=self.classes)
            else:
                tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        # Average feature importances across all trees
        all_importances = [t.feature_importances_ for t in self.trees if t.feature_importances_ is not None]
        if len(all_importances) > 0:
            self.feature_importances_ = np.mean(all_importances, axis=0)
            total = np.sum(self.feature_importances_)
            if total > 0:
                self.feature_importances_ /= total

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict labels/values using the ensemble."""
        if self.task == "classify":
            probas = self.predict_proba(X)
            return self.classes[np.argmax(probas, axis=1)]
        else:
            preds = np.array([tree.predict(X) for tree in self.trees])
            return np.mean(preds, axis=0)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities by averaging tree probabilities."""
        if self.task != "classify":
            raise ValueError("predict_proba is only available for classification.")

        all_probas = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(all_probas, axis=0)
