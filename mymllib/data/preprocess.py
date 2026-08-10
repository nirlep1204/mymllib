import numpy as np
import pandas as pd
from typing import Any, Tuple, Optional, Union


class Scaler:
    """Standardization (zero mean, unit variance): (x - mean) / std."""

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: Any) -> 'Scaler':
        """Compute the mean and std to be used for later scaling."""
        is_df = isinstance(X, pd.DataFrame)
        is_series = isinstance(X, pd.Series)
        vals = X.values if (is_df or is_series) else np.asarray(X)

        self.mean = np.mean(vals, axis=0)
        self.std = np.std(vals, axis=0)
        return self

    def fit_transform(self, X: Any) -> Any:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def transform(self, X: Any) -> Any:
        """Perform standardization by centering and scaling."""
        if self.mean is None or self.std is None:
            raise ValueError("Call .fit() or .fit_transform() before calling .transform().")

        is_df = isinstance(X, pd.DataFrame)
        is_series = isinstance(X, pd.Series)
        vals = X.values if (is_df or is_series) else np.asarray(X)

        std_safe = np.where(self.std == 0, 1.0, self.std)
        res = (vals - self.mean) / std_safe
        res = np.where(self.std == 0, 0.0, res)

        if is_df:
            return pd.DataFrame(res, index=X.index, columns=X.columns)
        if is_series:
            return pd.Series(res, index=X.index, name=X.name)
        return res

    def inverse_transform(self, X: Any) -> Any:
        """Scale back the data to the original representation."""
        if self.mean is None or self.std is None:
            raise ValueError("Call .fit() or .fit_transform() first.")

        is_df = isinstance(X, pd.DataFrame)
        is_series = isinstance(X, pd.Series)
        vals = X.values if (is_df or is_series) else np.asarray(X)

        res = vals * self.std + self.mean
        if is_df:
            return pd.DataFrame(res, index=X.index, columns=X.columns)
        if is_series:
            return pd.Series(res, index=X.index, name=X.name)
        return res


class Normalizer:
    """Min-max scaling to range [0, 1]: (x - min) / (max - min)."""

    def __init__(self) -> None:
        self.min: Optional[np.ndarray] = None
        self.max: Optional[np.ndarray] = None

    def fit(self, X: Any) -> 'Normalizer':
        """Compute the minimum and maximum to be used for later scaling."""
        is_df = isinstance(X, pd.DataFrame)
        is_series = isinstance(X, pd.Series)
        vals = X.values if (is_df or is_series) else np.asarray(X)

        self.min = np.min(vals, axis=0)
        self.max = np.max(vals, axis=0)
        return self

    def fit_transform(self, X: Any) -> Any:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def transform(self, X: Any) -> Any:
        """Scale features of X to range [0, 1]."""
        if self.min is None or self.max is None:
            raise ValueError("Call .fit() or .fit_transform() before calling .transform().")

        is_df = isinstance(X, pd.DataFrame)
        is_series = isinstance(X, pd.Series)
        vals = X.values if (is_df or is_series) else np.asarray(X)

        diff = self.max - self.min
        diff_safe = np.where(diff == 0, 1.0, diff)
        res = (vals - self.min) / diff_safe
        res = np.where(diff == 0, 0.0, res)

        if is_df:
            return pd.DataFrame(res, index=X.index, columns=X.columns)
        if is_series:
            return pd.Series(res, index=X.index, name=X.name)
        return res

    def inverse_transform(self, X: Any) -> Any:
        """Scale back the data to the original representation."""
        if self.min is None or self.max is None:
            raise ValueError("Call .fit() or .fit_transform() first.")

        is_df = isinstance(X, pd.DataFrame)
        is_series = isinstance(X, pd.Series)
        vals = X.values if (is_df or is_series) else np.asarray(X)

        res = vals * (self.max - self.min) + self.min
        if is_df:
            return pd.DataFrame(res, index=X.index, columns=X.columns)
        if is_series:
            return pd.Series(res, index=X.index, name=X.name)
        return res


class Encoder:
    """One-hot encoding for categorical columns."""

    def __init__(self) -> None:
        self.categories: Optional[list] = None

    def fit(self, data: pd.DataFrame, col: str) -> 'Encoder':
        """Discover unique categories in specified column."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame.")
        self.categories = list(pd.unique(data[col]))
        return self

    def fit_transform(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Fit to categorical column, then return one-hot encoded DataFrame."""
        return self.fit(data, col).transform(data, col)

    def transform(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Transform categorical column into one-hot binary columns."""
        if self.categories is None:
            raise ValueError("Call .fit() or .fit_transform() before calling .transform().")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame.")

        df = data.copy()
        col_idx = int(df.columns.get_loc(col)) # type: ignore

        encoded_cols = {}
        for cat in self.categories:
            encoded_cols[f"{col}_{cat}"] = (df[col] == cat).astype(int)

        encoded_df = pd.DataFrame(encoded_cols, index=df.index)

        left = df.iloc[:, :col_idx]
        right = df.iloc[:, col_idx + 1 :]

        return pd.concat([left, encoded_df, right], axis=1)


class LabelEncoder:
    """Convert labels to integers [0, 1, ..., K-1]."""

    def __init__(self) -> None:
        self.classes: Optional[np.ndarray] = None
        self.mapping: Optional[dict] = None

    def fit(self, y: Any) -> 'LabelEncoder':
        """Fit label encoder to classes."""
        y_arr = np.asarray(y)
        self.classes = np.unique(y_arr)
        self.mapping = {label: i for i, label in enumerate(self.classes)}
        return self

    def fit_transform(self, y: Any) -> np.ndarray:
        """Fit label encoder and return integer encoded array."""
        return self.fit(y).transform(y)

    def transform(self, y: Any) -> np.ndarray:
        """Transform labels to normalized integer encoding."""
        if self.mapping is None:
            raise ValueError("Call .fit() or .fit_transform() before calling .transform().")
        y_arr = np.asarray(y)
        return np.array([self.mapping[val] for val in y_arr])

    def inverse(self, y_encoded: Any) -> np.ndarray:
        """Transform integer labels back to original class names."""
        if self.classes is None:
            raise ValueError("Call .fit() or .fit_transform() first.")
        y_arr = np.asarray(y_encoded, dtype=int)
        return self.classes[y_arr]


def split(X: Any, y: Any, test: float = 0.2, seed: Optional[int] = None) -> Tuple[Any, Any, Any, Any]:
    """Train/test split."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    n_samples = len(X)
    if seed is not None:
        np.random.seed(seed)

    indices = np.random.permutation(n_samples)
    n_test = int(n_samples * test)
    split_idx = n_samples - n_test

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    def slice_data(data, idx):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.iloc[idx]
        return np.asarray(data)[idx]

    return slice_data(X, train_idx), slice_data(X, test_idx), slice_data(y, train_idx), slice_data(y, test_idx)


def xy(data: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features X and target y."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")
    if target not in data.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    X = data.drop(columns=[target])
    y = data[target]
    return X, y
