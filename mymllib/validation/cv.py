import copy
import numpy as np
import pandas as pd
from ..metrics.scores import accuracy, r2


def _to_numpy(a):
    if isinstance(a, (pd.DataFrame, pd.Series)):
        return a.to_numpy()
    return np.asarray(a)


class KFold:
    """K-Fold cross-validator."""

    def __init__(self, n_splits=5, shuffle=True, seed=None):
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.seed = seed

    def split(self, X, y=None):
        X_arr = _to_numpy(X)
        n_samples = len(X_arr)
        if self.n_splits <= 1 or self.n_splits > n_samples:
            raise ValueError(f"n_splits must be between 2 and {n_samples}.")

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            indices = rng.permutation(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0

        for f_size in fold_sizes:
            start, stop = current, current + f_size
            test_idx = indices[start:stop]
            train_idx = np.setdiff1d(indices, test_idx)
            yield train_idx, test_idx
            current = stop


class StratifiedKFold:
    """Stratified K-Fold cross-validator preserving class percentage per fold."""

    def __init__(self, n_splits=5, shuffle=True, seed=None):
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.seed = seed

    def split(self, X, y):
        if y is None:
            raise ValueError("StratifiedKFold requires target labels y.")
        y_arr = _to_numpy(y).ravel()
        n_samples = len(y_arr)

        if self.n_splits <= 1 or self.n_splits > n_samples:
            raise ValueError(f"n_splits must be between 2 and {n_samples}.")

        rng = np.random.default_rng(self.seed)
        unique_classes, y_inv = np.unique(y_arr, return_inverse=True)

        cls_indices = [np.where(y_inv == i)[0] for i in range(len(unique_classes))]
        if self.shuffle:
            cls_indices = [rng.permutation(idx) for idx in cls_indices]

        # Distribute each class samples across folds
        test_folds = [[] for _ in range(self.n_splits)]
        for idxs in cls_indices:
            splits = np.array_split(idxs, self.n_splits)
            for fold_i, split_idxs in enumerate(splits):
                test_folds[fold_i].extend(split_idxs)

        all_indices = np.arange(n_samples)
        for fold_test in test_folds:
            test_idx = np.array(fold_test, dtype=int)
            train_idx = np.setdiff1d(all_indices, test_idx)
            yield train_idx, test_idx


class LeaveOneOut:
    """Leave-One-Out (LOOCV) cross-validator where test set is 1 sample per fold."""

    def __init__(self):
        pass

    def split(self, X, y=None):
        X_arr = _to_numpy(X)
        n_samples = len(X_arr)
        indices = np.arange(n_samples)
        for i in range(n_samples):
            test_idx = np.array([i])
            train_idx = np.delete(indices, i)
            yield train_idx, test_idx


# Friendly alias
LOOCV = LeaveOneOut


class ShuffleSplit:
    """Random permutation train/test cross-validator."""

    def __init__(self, n_splits=5, test_size=0.2, seed=None):
        self.n_splits = int(n_splits)
        self.test_size = float(test_size)
        self.seed = seed

    def split(self, X, y=None):
        X_arr = _to_numpy(X)
        n_samples = len(X_arr)
        n_test = max(1, int(n_samples * self.test_size))

        rng = np.random.default_rng(self.seed)
        indices = np.arange(n_samples)

        for _ in range(self.n_splits):
            shuffled = rng.permutation(indices)
            test_idx = shuffled[:n_test]
            train_idx = shuffled[n_test:]
            yield train_idx, test_idx


class TimeSeriesSplit:
    """Time-series forward chaining cross-validator without data leakage."""

    def __init__(self, n_splits=5):
        self.n_splits = int(n_splits)

    def split(self, X, y=None):
        X_arr = _to_numpy(X)
        n_samples = len(X_arr)
        if self.n_splits + 1 > n_samples:
            raise ValueError(f"Too many splits for {n_samples} samples.")

        test_size = n_samples // (self.n_splits + 1)
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            train_end = (i + 1) * test_size
            test_end = train_end + test_size
            if i == self.n_splits - 1:
                test_end = n_samples

            yield indices[:train_end], indices[train_end:test_end]


def cross_validate(model, X, y, cv=5, folds=None, metric=None, seed=None):
    """Perform cross-validation on a model using KFold, StratifiedKFold, or any splitter.

    Parameters:
    - model: Any estimator instance (e.g. ml.Linear, ml.SVM, ml.Forest)
    - X: Feature matrix (numpy array or pandas DataFrame)
    - y: Target labels or values
    - cv: Integer (number of folds) or CV splitter instance (e.g. KFold, StratifiedKFold, LeaveOneOut)
    - folds: Optional alias for cv (e.g. folds=5)
    - metric: Scoring function (e.g. ml.accuracy, ml.mse, ml.f1, ml.r2). Defaults to accuracy for classification, r2 for regression.
    - seed: Random seed for shuffling

    Returns:
    - numpy array of test scores for each fold
    """
    if folds is not None:
        cv = folds

    X_arr = _to_numpy(X)
    y_arr = _to_numpy(y).ravel()
    n_samples = len(y_arr)

    # Detect classification vs regression
    unique_vals = np.unique(y_arr)
    is_classification = len(unique_vals) <= 20 and not np.issubdtype(y_arr.dtype, np.floating)

    if metric is None:
        metric = accuracy if is_classification else r2

    # Determine CV splitter
    if isinstance(cv, int):
        if is_classification:
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, seed=seed)
        else:
            splitter = KFold(n_splits=cv, shuffle=True, seed=seed)
    elif hasattr(cv, "split"):
        splitter = cv
    else:
        raise ValueError("cv must be an integer or a splitter object with a .split() method.")

    scores = []
    for train_idx, test_idx in splitter.split(X_arr, y_arr):
        X_train, y_train = X_arr[train_idx], y_arr[train_idx]
        X_test, y_test = X_arr[test_idx], y_arr[test_idx]

        model_clone = copy.deepcopy(model)
        model_clone.fit(X_train, y_train)
        preds = model_clone.predict(X_test)

        score = metric(y_test, preds)
        scores.append(float(score))

    return np.array(scores)
