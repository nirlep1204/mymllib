import copy
import itertools
import numpy as np
from typing import Any, Dict, List, Optional, Iterator, Callable
from .cv import cross_validate


class GridSearchCV:
    """Exhaustive hyperparameter grid search using cross-validation."""

    def __init__(self, model: Any, param_grid: Dict[str, List[Any]], cv: Any = 5, metric: Optional[Callable] = None, seed: Optional[int] = None) -> None:
        """Initialize Grid Search.

        Parameters:
        - model: Base estimator model template (e.g. ml.SVM(), ml.Linear(), ml.Forest())
        - param_grid: Dictionary with parameter names as keys and lists of parameter settings to try.
        - cv: Number of folds (default 5) or CV splitter instance.
        - metric: Evaluation metric function (e.g. ml.accuracy, ml.mse, ml.r2).
        - seed: Random seed.
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.metric = metric
        self.seed = seed

        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_ = -np.inf
        self.best_model_: Optional[Any] = None
        self.cv_results_: List[Dict[str, Any]] = []

    def _generate_candidates(self) -> Iterator[Dict[str, Any]]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def fit(self, X: Any, y: Any) -> 'GridSearchCV':
        """Fit all parameter combinations and find the best model."""
        candidates = list(self._generate_candidates())
        self.cv_results_ = []
        best_score = -np.inf
        best_params = None

        for params in candidates:
            # Clone model and apply parameters
            candidate_model = copy.deepcopy(self.model)
            for k, v in params.items():
                setattr(candidate_model, k, v)

            scores = cross_validate(candidate_model, X, y, cv=self.cv, metric=self.metric, seed=self.seed)
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))

            self.cv_results_.append({
                "params": params,
                "mean_score": mean_score,
                "std_score": std_score,
                "scores": scores.tolist()
            })

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.best_score_ = best_score
        self.best_params_ = best_params

        # Refit best model on full dataset
        self.best_model_ = copy.deepcopy(self.model)
        if self.best_params_ is not None:
            for k, v in self.best_params_.items():
                setattr(self.best_model_, k, v)
        if hasattr(self.best_model_, "fit"):
            self.best_model_.fit(X, y)

        return self

    def predict(self, X: Any) -> Any:
        """Predict using the best fitted model."""
        if self.best_model_ is None:
            raise ValueError("This GridSearchCV instance is not fitted yet. Call .fit() first.")
        return self.best_model_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Predict probabilities using the best fitted model."""
        if self.best_model_ is None:
            raise ValueError("This GridSearchCV instance is not fitted yet. Call .fit() first.")
        if hasattr(self.best_model_, "predict_proba"):
            return self.best_model_.predict_proba(X)
        raise AttributeError(f"{type(self.best_model_).__name__} does not support predict_proba.")

    def summary(self) -> None:
        """Print a summary table of hyperparameter search results."""
        sorted_results = sorted(self.cv_results_, key=lambda r: r["mean_score"], reverse=True)
        print("=" * 65)
        print("                 GridSearchCV Results               ".center(65))
        print("=" * 65)
        print(f"Best Score : {self.best_score_:.4f}")
        print(f"Best Params: {self.best_params_}")
        print("-" * 65)
        print(f"{'Rank':<5} | {'Mean Score':<12} | {'Std Dev':<10} | {'Parameters'}")
        print("-" * 65)
        for rank, res in enumerate(sorted_results, start=1):
            p_str = ", ".join(f"{k}={v}" for k, v in res["params"].items())
            print(f"{rank:<5} | {res['mean_score']:<12.4f} | {res['std_score']:<10.4f} | {p_str}")
        print("=" * 65)


class RandomSearchCV:
    """Randomized hyperparameter search using cross-validation."""

    def __init__(self, model: Any, param_distributions: Dict[str, List[Any]], n_iter: int = 10, cv: Any = 5, metric: Optional[Callable] = None, seed: Optional[int] = None) -> None:
        """Initialize Randomized Search.

        Parameters:
        - model: Base estimator model template.
        - param_distributions: Dictionary with parameter names as keys and lists of options to sample from.
        - n_iter: Number of parameter settings sampled (default 10).
        - cv: Number of folds (default 5) or CV splitter instance.
        - metric: Evaluation metric function.
        - seed: Random seed.
        """
        self.model = model
        self.param_distributions = param_distributions
        self.n_iter = int(n_iter)
        self.cv = cv
        self.metric = metric
        self.seed = seed

        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_ = -np.inf
        self.best_model_: Optional[Any] = None
        self.cv_results_: List[Dict[str, Any]] = []

    def _sample_candidates(self) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        keys = list(self.param_distributions.keys())
        sampled: List[Dict[str, Any]] = []
        seen = set()

        # Calculate max possible combinations
        max_possible = 1
        for v in self.param_distributions.values():
            max_possible *= len(v)

        target_iter = min(self.n_iter, max_possible)

        while len(sampled) < target_iter:
            combo = {}
            for k in keys:
                options = self.param_distributions[k]
                combo[k] = options[rng.integers(0, len(options))]
            
            combo_tuple = tuple(sorted((k, str(v)) for k, v in combo.items()))
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                sampled.append(combo)

        return sampled

    def fit(self, X: Any, y: Any) -> 'RandomSearchCV':
        """Fit sampled parameter combinations and find the best model."""
        candidates = self._sample_candidates()
        self.cv_results_ = []
        best_score = -np.inf
        best_params = None

        for params in candidates:
            candidate_model = copy.deepcopy(self.model)
            for k, v in params.items():
                setattr(candidate_model, k, v)

            scores = cross_validate(candidate_model, X, y, cv=self.cv, metric=self.metric, seed=self.seed)
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))

            self.cv_results_.append({
                "params": params,
                "mean_score": mean_score,
                "std_score": std_score,
                "scores": scores.tolist()
            })

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.best_score_ = best_score
        self.best_params_ = best_params

        # Refit best model on full dataset
        self.best_model_ = copy.deepcopy(self.model)
        if self.best_params_ is not None:
            for k, v in self.best_params_.items():
                setattr(self.best_model_, k, v)
        if hasattr(self.best_model_, "fit"):
            self.best_model_.fit(X, y)

        return self

    def predict(self, X: Any) -> Any:
        """Predict using the best fitted model."""
        if self.best_model_ is None:
            raise ValueError("This RandomSearchCV instance is not fitted yet. Call .fit() first.")
        return self.best_model_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Predict probabilities using the best fitted model."""
        if self.best_model_ is None:
            raise ValueError("This RandomSearchCV instance is not fitted yet. Call .fit() first.")
        if hasattr(self.best_model_, "predict_proba"):
            return self.best_model_.predict_proba(X)
        raise AttributeError(f"{type(self.best_model_).__name__} does not support predict_proba.")

    def summary(self) -> None:
        """Print a summary table of hyperparameter search results."""
        sorted_results = sorted(self.cv_results_, key=lambda r: r["mean_score"], reverse=True)
        print("=" * 65)
        print("                RandomSearchCV Results              ".center(65))
        print("=" * 65)
        print(f"Best Score : {self.best_score_:.4f}")
        print(f"Best Params: {self.best_params_}")
        print("-" * 65)
        print(f"{'Rank':<5} | {'Mean Score':<12} | {'Std Dev':<10} | {'Parameters'}")
        print("-" * 65)
        for rank, res in enumerate(sorted_results, start=1):
            p_str = ", ".join(f"{k}={v}" for k, v in res["params"].items())
            print(f"{rank:<5} | {res['mean_score']:<12.4f} | {res['std_score']:<10.4f} | {p_str}")
        print("=" * 65)
