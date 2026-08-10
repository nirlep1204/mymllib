import numpy as np
import pandas as pd
from typing import Union, Any


def _to_numpy(X):
    """Convert input to numpy array."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy()
    return np.asarray(X)


def _log_to_prob(log_posteriors):
    """Convert log posteriors to normalized probabilities safely."""
    # subtract max for numerical stability before exp
    max_log = np.max(log_posteriors, axis=1, keepdims=True)
    exp_prob = np.exp(log_posteriors - max_log)
    return exp_prob / np.sum(exp_prob, axis=1, keepdims=True)


class GaussianNB:
    def fit(self, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any]) -> "GaussianNB":
        """Fit Gaussian Naive Bayes model."""
        X = _to_numpy(X)
        y = _to_numpy(y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.priors = np.zeros(n_classes, dtype=np.float64)
        self.means = np.zeros((n_classes, n_features), dtype=np.float64)
        self.vars = np.zeros((n_classes, n_features), dtype=np.float64)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[i] = len(X_c) / len(X)
            self.means[i, :] = np.mean(X_c, axis=0)
            self.vars[i, :] = np.var(X_c, axis=0)

        return self

    def _log_likelihood(self, x, mean, var):
        """Compute log likelihood of data given mean and variance."""
        eps = 1e-9
        return -0.5 * np.log(2 * np.pi * (var + eps)) - 0.5 * ((x - mean) ** 2) / (var + eps)

    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Predict class labels for samples in X."""
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]

    def predict_proba(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Predict class probabilities for samples in X."""
        X = _to_numpy(X)
        log_posteriors = np.zeros((len(X), len(self.classes)))

        for i, _ in enumerate(self.classes):
            prior = np.log(self.priors[i])
            likelihood = np.sum(self._log_likelihood(X, self.means[i], self.vars[i]), axis=1)
            log_posteriors[:, i] = prior + likelihood

        return _log_to_prob(log_posteriors)


class MultinomialNB:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any]) -> "MultinomialNB":
        """Fit Multinomial Naive Bayes model."""
        X = _to_numpy(X)
        y = _to_numpy(y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.log_priors = np.zeros(n_classes, dtype=np.float64)
        self.feature_log_prob = np.zeros((n_classes, n_features), dtype=np.float64)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.log_priors[i] = np.log(len(X_c) / len(X))

            feature_count = np.sum(X_c, axis=0)
            total_count = np.sum(feature_count)

            # apply Laplace smoothing
            smoothed_count = feature_count + self.alpha
            smoothed_total = total_count + self.alpha * n_features
            
            self.feature_log_prob[i, :] = np.log(smoothed_count / smoothed_total)

        return self

    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Predict class labels for samples in X."""
        X = _to_numpy(X)
        log_posteriors = self.log_priors + X @ self.feature_log_prob.T
        return self.classes[np.argmax(log_posteriors, axis=1)]

    def predict_proba(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Predict class probabilities for samples in X."""
        X = _to_numpy(X)
        log_posteriors = self.log_priors + X @ self.feature_log_prob.T
        return _log_to_prob(log_posteriors)


class BernoulliNB:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any]) -> "BernoulliNB":
        """Fit Bernoulli Naive Bayes model."""
        X = _to_numpy(X)
        y = _to_numpy(y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.log_priors = np.zeros(n_classes, dtype=np.float64)
        self.feature_log_prob = np.zeros((n_classes, n_features), dtype=np.float64)
        self.feature_log_neg_prob = np.zeros((n_classes, n_features), dtype=np.float64)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            class_count = len(X_c)
            self.log_priors[i] = np.log(class_count / len(X))

            feature_count = np.sum(X_c, axis=0)

            # prob of feature being 1
            prob = (feature_count + self.alpha) / (class_count + 2 * self.alpha)
            
            self.feature_log_prob[i, :] = np.log(prob)
            self.feature_log_neg_prob[i, :] = np.log(1.0 - prob)

        return self

    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Predict class labels for samples in X."""
        X = _to_numpy(X)
        log_posteriors = self.log_priors + X @ self.feature_log_prob.T + (1 - X) @ self.feature_log_neg_prob.T
        return self.classes[np.argmax(log_posteriors, axis=1)]

    def predict_proba(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Predict class probabilities for samples in X."""
        X = _to_numpy(X)
        log_posteriors = self.log_priors + X @ self.feature_log_prob.T + (1 - X) @ self.feature_log_neg_prob.T
        return _log_to_prob(log_posteriors)
