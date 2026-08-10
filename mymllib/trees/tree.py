import numpy as np
import pandas as pd
from typing import Optional, Any, Union, Dict, Tuple


class Node:
    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None, left: Optional['Node'] = None, right: Optional['Node'] = None, value: Optional[Any] = None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        return self.value is not None


class Tree:
    def __init__(self, task: str = "classify", max_depth: int = 10, min_samples_split: int = 2, min_impurity_decrease: float = 1e-7, max_features: Union[str, float, int, None] = None, seed: Optional[int] = None) -> None:
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.seed = seed
        self.root: Optional['Node'] = None
        self.classes: Optional[np.ndarray] = None
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: Union[pd.DataFrame, pd.Series, np.ndarray], y: Union[pd.DataFrame, pd.Series, np.ndarray], classes: Optional[np.ndarray] = None) -> "Tree":
        """Build decision tree recursively."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values # type: ignore
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values # type: ignore

        X = np.asarray(X)
        y = np.asarray(y)

        if self.task == "classify":
            if classes is not None:
                self.classes = np.asarray(classes)
            else:
                self.classes = np.unique(y)

        self.n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features)
        self.root = self._build_tree(X, y, depth=0)

        # Normalize feature importances
        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp

        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            return Node(value=self._leaf_value(y))

        feat_idxs = self._get_features(n_features)
        best_feat, best_thresh, best_gain = self._best_split(X, y, feat_idxs)

        if best_gain < self.min_impurity_decrease or best_feat is None:
            return Node(value=self._leaf_value(y))

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return Node(value=self._leaf_value(y))

        # Accumulate weighted impurity decrease for feature importance
        self.feature_importances_[best_feat] += best_gain * n_samples

        left_node = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_node = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, left=left_node, right=right_node)

    def _get_features(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)

        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                n_select = int(np.sqrt(n_features))
            elif self.max_features == "log2":
                n_select = int(np.log2(n_features))
            else:
                n_select = n_features
        elif isinstance(self.max_features, float):
            n_select = int(self.max_features * n_features)
        else:
            n_select = self.max_features

        n_select = max(1, min(n_features, n_select))
        return self._rng.choice(n_features, n_select, replace=False)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        best_feat = None
        best_thresh = None
        parent_impurity = self._impurity(y)

        for feat in feat_idxs:
            X_col = X[:, feat]
            thresholds = np.unique(X_col)

            for thresh in thresholds:
                gain = self._info_gain(y, X_col, thresh, parent_impurity)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    def _info_gain(self, y, X_col, thresh, parent_impurity):
        left_idx = X_col <= thresh
        right_idx = X_col > thresh

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return 0

        n = len(y)
        n_l, n_r = np.sum(left_idx), np.sum(right_idx)

        child_impurity = (n_l / n) * self._impurity(y[left_idx]) + (n_r / n) * self._impurity(y[right_idx])
        return parent_impurity - child_impurity

    def _impurity(self, y):
        if self.task == "classify":
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return 1 - np.sum(probs ** 2)
        else:
            if len(y) == 0:
                return 0
            return np.var(y)

    def _leaf_value(self, y):
        if self.task == "classify":
            counts = {c: 0 for c in self.classes}
            unique, counts_arr = np.unique(y, return_counts=True)
            for val, count in zip(unique, counts_arr):
                counts[val] = count

            probs = {c: counts[c] / len(y) for c in self.classes}
            majority_class = unique[np.argmax(counts_arr)]

            return {"class": majority_class, "probs": probs}
        else:
            return np.mean(y)

    def predict(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """Predict labels/values for samples."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values # type: ignore
        X = np.asarray(X)

        preds_list = [self._traverse_tree(x, self.root) for x in X]
        if self.task == "classify":
            preds = np.array([p["class"] for p in preds_list])
        else:
            preds = np.array(preds_list)

        return preds

    def predict_proba(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        if self.task != "classify":
            raise ValueError("predict_proba is only available for classification.")

        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values # type: ignore
        X = np.asarray(X)

        leaf_vals = [self._traverse_tree(x, self.root) for x in X]
        probs = []
        for val in leaf_vals:
            probs.append([val["probs"][c] for c in self.classes]) # type: ignore

        return np.array(probs)

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
