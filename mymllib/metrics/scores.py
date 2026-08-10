from typing import Any, Tuple, Optional
import numpy as np
import pandas as pd


def _to_numpy(a: Any) -> np.ndarray:
    if isinstance(a, (pd.DataFrame, pd.Series)):
        return a.to_numpy()
    return np.asarray(a)


def mse(y: Any, pred: Any) -> float:
    """Mean Squared Error."""
    y = _to_numpy(y).ravel()
    pred = _to_numpy(pred).ravel()
    return float(np.mean((y - pred) ** 2))


def rmse(y: Any, pred: Any) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y, pred)))


def mae(y: Any, pred: Any) -> float:
    """Mean Absolute Error."""
    y = _to_numpy(y).ravel()
    pred = _to_numpy(pred).ravel()
    return float(np.mean(np.abs(y - pred)))


def r2(y: Any, pred: Any) -> float:
    """R-squared (coefficient of determination)."""
    y = _to_numpy(y).ravel()
    pred = _to_numpy(pred).ravel()
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def accuracy(y: Any, pred: Any) -> float:
    """Classification accuracy."""
    y = _to_numpy(y).ravel()
    pred = _to_numpy(pred).ravel()
    return float(np.mean(y == pred))


def precision(y: Any, pred: Any, average: str = "macro", pos_label: int = 1) -> float:
    """Precision score (binary or macro)."""
    y = _to_numpy(y).ravel()
    pred = _to_numpy(pred).ravel()
    classes = np.unique(np.concatenate([y, pred]))

    if len(classes) == 2 and (pos_label in classes) and average == "binary":
        tp = np.sum((y == pos_label) & (pred == pos_label))
        fp = np.sum((y != pos_label) & (pred == pos_label))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    precisions = []
    for c in classes:
        tp = np.sum((y == c) & (pred == c))
        fp = np.sum((y != c) & (pred == c))
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return float(np.mean(precisions))


def recall(y: Any, pred: Any, average: str = "macro", pos_label: int = 1) -> float:
    """Recall score (binary or macro)."""
    y = _to_numpy(y).ravel()
    pred = _to_numpy(pred).ravel()
    classes = np.unique(np.concatenate([y, pred]))

    if len(classes) == 2 and (pos_label in classes) and average == "binary":
        tp = np.sum((y == pos_label) & (pred == pos_label))
        fn = np.sum((y == pos_label) & (pred != pos_label))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    recalls = []
    for c in classes:
        tp = np.sum((y == c) & (pred == c))
        fn = np.sum((y == c) & (pred != c))
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(recalls))


def f1(y: Any, pred: Any, average: str = "macro", pos_label: int = 1) -> float:
    """F1 score."""
    p = precision(y, pred, average=average, pos_label=pos_label)
    r = recall(y, pred, average=average, pos_label=pos_label)
    if p + r == 0:
        return 0.0
    return float(2.0 * (p * r) / (p + r))


def confusion(y: Any, pred: Any) -> np.ndarray:
    """Confusion matrix where rows are true labels and columns are predicted labels."""
    y = _to_numpy(y).ravel()
    pred = _to_numpy(pred).ravel()
    classes = np.unique(np.concatenate([y, pred]))
    k = len(classes)
    matrix = np.zeros((k, k), dtype=int)
    class_map = {c: i for i, c in enumerate(classes)}

    for true_val, pred_val in zip(y, pred):
        matrix[class_map[true_val], class_map[pred_val]] += 1
    return matrix


def roc_curve(y_true: Any, y_score: Any, pos_label: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Receiver Operating Characteristic (ROC) curve.

    Parameters:
    - y_true: True binary labels (0/1 or class names)
    - y_score: Target probabilities or decision function values
    - pos_label: Label of the positive class

    Returns:
    - fpr: Increasing False Positive Rates (numpy array)
    - tpr: Increasing True Positive Rates (numpy array)
    - thresholds: Decreasing decision thresholds (numpy array)
    """
    y_arr = _to_numpy(y_true).ravel()
    score_arr = _to_numpy(y_score).ravel()

    classes = np.unique(y_arr)
    if len(classes) != 2:
        raise ValueError("ROC curve is only defined for binary classification.")

    target_pos = pos_label if pos_label in classes else classes[1]
    y_bin = (y_arr == target_pos).astype(int)
    n_pos = int(np.sum(y_bin == 1))
    n_neg = int(np.sum(y_bin == 0))

    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true must contain both positive and negative samples.")

    # Sort descending by score
    desc_idx = np.argsort(score_arr)[::-1]
    y_sorted = y_bin[desc_idx]
    score_sorted = score_arr[desc_idx]

    distinct_mask = np.r_[True, score_sorted[1:] != score_sorted[:-1]]
    threshold_idxs = np.where(distinct_mask)[0]

    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[score_sorted[0] + 1.0, score_sorted[threshold_idxs]]

    tpr = tps / n_pos
    fpr = fps / n_neg

    return fpr, tpr, thresholds


def auc(x: Any, y: Any) -> float:
    """Compute Area Under the Curve using trapezoidal rule."""
    x_arr = _to_numpy(x).ravel()
    y_arr = _to_numpy(y).ravel()
    if len(x_arr) < 2:
        return 0.0
    dx = np.diff(x_arr)
    return float(np.sum(dx * (y_arr[:-1] + y_arr[1:]) / 2.0))


def roc_auc_score(y_true: Any, y_score: Any, pos_label: int = 1) -> float:
    """Compute Area Under the Receiver Operating Characteristic (ROC AUC) Score."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    return auc(fpr, tpr)


def wcss(X: Any, labels: Any, centers: Optional[Any] = None) -> float:
    """Within-cluster sum of squares."""
    X = _to_numpy(X)
    labels = _to_numpy(labels).ravel()
    unique_labels = np.unique(labels)

    total_wcss = 0.0
    for i, c in enumerate(unique_labels):
        cluster_points = X[labels == c]
        if len(cluster_points) == 0:
            continue
        if centers is not None:
            center = centers[i] if i < len(centers) else np.mean(cluster_points, axis=0)
        else:
            center = np.mean(cluster_points, axis=0)
        total_wcss += np.sum((cluster_points - center) ** 2)
    return float(total_wcss)


def silhouette(X: Any, labels: Any) -> float:
    """Silhouette score for clustering quality."""
    X = _to_numpy(X)
    labels = _to_numpy(labels).ravel()
    unique_labels = np.unique(labels)
    n_samples = X.shape[0]

    if len(unique_labels) <= 1 or len(unique_labels) >= n_samples:
        return 0.0

    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    dists = np.sqrt(np.maximum(0.0, X_sq + X_sq.T - 2.0 * X @ X.T))

    s_scores = np.zeros(n_samples)
    for i in range(n_samples):
        c_i = labels[i]
        same_cluster = (labels == c_i)
        if np.sum(same_cluster) <= 1:
            s_scores[i] = 0.0
            continue

        a_i = np.sum(dists[i, same_cluster]) / (np.sum(same_cluster) - 1)

        b_i = np.inf
        for other_c in unique_labels:
            if other_c == c_i:
                continue
            other_cluster = (labels == other_c)
            if np.sum(other_cluster) == 0:
                continue
            mean_dist = np.mean(dists[i, other_cluster])
            if mean_dist < b_i:
                b_i = mean_dist

        max_ab = max(a_i, b_i)
        s_scores[i] = (b_i - a_i) / max_ab if max_ab > 0 else 0.0

    return float(np.mean(s_scores))
