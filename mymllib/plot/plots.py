import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..metrics.scores import confusion


def _to_numpy(a):
    if a is None:
        return None
    if isinstance(a, (pd.DataFrame, pd.Series)):
        return a.to_numpy()
    return np.asarray(a)


def scatter(x, y, c=None, color="royalblue", label=None, title="Scatter Plot", xlabel="X", ylabel="Y", show=True):
    """Scatter plot with optional color-coding, custom colors, and legend labels."""
    x_arr = _to_numpy(x)
    y_arr = _to_numpy(y)

    if x_arr.ndim > 1 and x_arr.shape[1] == 1:
        x_arr = x_arr.ravel()
    if y_arr.ndim > 1 and y_arr.shape[1] == 1:
        y_arr = y_arr.ravel()

    # Check if a figure already exists, if not create one
    if not plt.get_fignums():
        plt.figure(figsize=(7, 5))

    if c is not None:
        c_arr = _to_numpy(c)
        plt.scatter(x_arr, y_arr, c=c_arr, cmap="viridis", edgecolors="k", alpha=0.8, label=label)
    else:
        plt.scatter(x_arr, y_arr, color=color, edgecolors="k", alpha=0.8, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)

    if label is not None:
        plt.legend()

    plt.tight_layout()
    if show:
        plt.show()


def line(x, y, color="crimson", label=None, equation=None, title="Line Plot", xlabel="X", ylabel="Y", show=True):
    """Line plot with color selection, legend label, and optional equation printing / annotation."""
    x_arr = _to_numpy(x)
    y_arr = _to_numpy(y)

    if x_arr.ndim > 1 and x_arr.shape[1] == 1:
        x_arr = x_arr.ravel()
    if y_arr.ndim > 1 and y_arr.shape[1] == 1:
        y_arr = y_arr.ravel()

    # Check if a figure already exists, if not create one
    if not plt.get_fignums():
        plt.figure(figsize=(7, 5))

    # Sort by x for clean contiguous line drawing
    sort_idx = np.argsort(x_arr)
    x_sorted = x_arr[sort_idx]
    y_sorted = y_arr[sort_idx]

    eq_str = None

    # Handle equation argument
    if equation is not None:
        if isinstance(equation, str):
            eq_str = equation
        elif hasattr(equation, "w") and hasattr(equation, "b"):
            # Model instance passed (e.g. Linear)
            w = np.asarray(equation.w).ravel()
            b = float(equation.b) if equation.b is not None else 0.0
            if len(w) == 1:
                eq_str = f"y = {w[0]:.4f} * x + ({b:.4f})"
            else:
                terms = [f"{w[i]:.4f}*x{i+1}" for i in range(len(w))]
                eq_str = "y = " + " + ".join(terms) + f" + ({b:.4f})"
        elif equation is True:
            # Auto-compute line equation from (x, y)
            x_mean = np.mean(x_sorted)
            y_mean = np.mean(y_sorted)
            denom = np.sum((x_sorted - x_mean) ** 2)
            if denom > 1e-12:
                m = np.sum((x_sorted - x_mean) * (y_sorted - y_mean)) / denom
                b = y_mean - m * x_mean
                eq_str = f"y = {m:.4f} * x + ({b:.4f})"

        if eq_str:
            print(f"Fitted Line Equation: {eq_str}")
            if label:
                label = f"{label} ({eq_str})"
            else:
                label = eq_str

    plt.plot(x_sorted, y_sorted, linewidth=2, color=color, marker="o", markersize=4, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)

    if label is not None or eq_str is not None:
        plt.legend()

    plt.tight_layout()
    if show:
        plt.show()


def loss_plot(losses, color="crimson", title="Training Loss Curve", xlabel="Iteration / Epoch", ylabel="Loss", show=True):
    """Plot training loss over iterations/epochs.
    
    Parameters:
    - losses: Model instance (e.g. model) or list/array of recorded losses (e.g. model.losses)
    - color: Plot curve color (e.g. 'crimson', 'blue', 'purple', 'teal')
    - title: Plot title string
    - xlabel: X-axis label
    - ylabel: Y-axis label
    - show: Whether to call plt.show() immediately
    """
    # If a model instance is passed directly, extract its losses attribute
    if hasattr(losses, "losses"):
        losses = losses.losses

    if losses is None or len(losses) == 0:
        print("[mymllib.loss_plot] Warning: No loss history found. Note that closed-form methods (e.g. method='normal') compute exact solutions without gradient descent iterations.")
        return

    loss_arr = np.asarray(losses, dtype=float).ravel()
    if len(loss_arr) == 0:
        print("[mymllib.loss_plot] Warning: Loss array is empty.")
        return

    # Check for non-finite values
    finite_mask = np.isfinite(loss_arr)
    if not np.any(finite_mask):
        print("[mymllib.loss_plot] Error: Loss values are all NaN or Inf. Try lowering the learning rate (lr).")
        return

    print(f"Training Loss: Initial = {loss_arr[0]:.6f} -> Final = {loss_arr[-1]:.6f} ({len(loss_arr)} recorded steps)")

    plt.figure(figsize=(7, 5))
    x_axis = np.arange(0, len(loss_arr)) if len(loss_arr) > 1 else np.array([1])
    marker = "o" if len(loss_arr) <= 40 else None
    
    plt.plot(x_axis, loss_arr, linewidth=2, color=color, marker=marker, markersize=4, label="Loss")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


def hist(data, col=None, bins=20, color="skyblue", title=None, xlabel=None, ylabel="Frequency", show=True):
    """Histogram plot."""
    if isinstance(data, pd.DataFrame) and col is not None:
        vals = data[col].dropna().values
        title = title or f"Histogram of {col}"
        xlabel = xlabel or str(col)
    else:
        vals = _to_numpy(data).ravel()
        vals = vals[~np.isnan(vals)]
        title = title or "Histogram"
        xlabel = xlabel or "Value"

    plt.figure(figsize=(7, 5))
    plt.hist(vals, bins=bins, edgecolor="black", alpha=0.75, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if show:
        plt.show()


def boundary(model, X, y, title="Decision Boundary", resolution=200, cmap="coolwarm", show=True):
    """Plot 2D decision boundary for a classifier."""
    X_arr = _to_numpy(X)
    y_arr = _to_numpy(y).ravel()

    if X_arr.shape[1] != 2:
        raise ValueError("Boundary plot requires 2D features (X.shape[1] == 2).")

    x_min, x_max = X_arr[:, 0].min() - 1.0, X_arr[:, 0].max() + 1.0
    y_min, y_max = X_arr[:, 1].min() - 1.0, X_arr[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    preds = model.predict(grid_points)

    # Convert non-numeric classes if needed
    classes = np.unique(y_arr)
    class_map = {c: i for i, c in enumerate(classes)}
    preds_num = np.array([class_map[p] for p in preds]).reshape(xx.shape)
    y_num = np.array([class_map[c] for c in y_arr])

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds_num, alpha=0.3, cmap=cmap)
    plt.scatter(
        X_arr[:, 0], X_arr[:, 1], c=y_num, cmap=cmap, edgecolors="k", alpha=0.9
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if show:
        plt.show()


def confusion_plot(y, pred, cmap="Blues", title="Confusion Matrix", show=True):
    """Plot confusion matrix heatmap."""
    cm = confusion(y, pred)
    y_arr = _to_numpy(y).ravel()
    pred_arr = _to_numpy(pred).ravel()
    classes = np.unique(np.concatenate([y_arr, pred_arr]))

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if show:
        plt.show()


def dendrogram(model, color="purple", title="Hierarchical Dendrogram", show=True):
    """Plot simple dendrogram from a fitted Hierarchical clustering model."""
    if hasattr(model, "dendrogram_data"):
        history = model.dendrogram_data()
    elif isinstance(model, list):
        history = model
    else:
        raise ValueError("Expected Hierarchical model or list of merge records.")

    plt.figure(figsize=(8, 5))
    distances = [rec[2] for rec in history] if history else [0]
    plt.plot(range(1, len(distances) + 1), distances, marker="s", color=color)
    plt.title(title)
    plt.xlabel("Merge Step")
    plt.ylabel("Merge Distance")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if show:
        plt.show()


def pca_plot(X, y=None, color="teal", cmap="tab10", title="PCA Projection", show=True):
    """Project data to 2D using PCA and scatter plot."""
    from ..reduce.pca import PCA

    X_arr = _to_numpy(X)
    if X_arr.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_arr)
    else:
        X_2d = X_arr

    plt.figure(figsize=(7, 5))
    if y is not None:
        y_arr = _to_numpy(y).ravel()
        plt.scatter(
            X_2d[:, 0], X_2d[:, 1], c=y_arr, cmap=cmap, edgecolors="k", alpha=0.8
        )
    else:
        plt.scatter(X_2d[:, 0], X_2d[:, 1], edgecolors="k", alpha=0.8, color=color)

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if show:
        plt.show()


def cluster_plot(X, labels, centers=None, cmap="rainbow", title="Cluster Assignments", show=True):
    """Scatter plot of 2D cluster assignments with optional cluster centers."""
    X_arr = _to_numpy(X)
    labels_arr = _to_numpy(labels).ravel()

    if X_arr.shape[1] > 2:
        from ..reduce.pca import PCA

        pca = PCA(n_components=2)
        X_plot = pca.fit_transform(X_arr)
        if centers is not None:
            centers_plot = pca.transform(centers)
        else:
            centers_plot = None
    else:
        X_plot = X_arr
        centers_plot = _to_numpy(centers) if centers is not None else None

    plt.figure(figsize=(7, 5))
    plt.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        c=labels_arr,
        cmap=cmap,
        edgecolors="k",
        alpha=0.75,
    )

    if centers_plot is not None:
        plt.scatter(
            centers_plot[:, 0],
            centers_plot[:, 1],
            c="black",
            s=200,
            marker="X",
            label="Centroids",
        )
        plt.legend()

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if show:
        plt.show()


def roc_plot(y_true, y_score, pos_label=1, color="crimson", title="Receiver Operating Characteristic (ROC)", show=True):
    """Plot ROC Curve with AUC score calculation and diagonal baseline."""
    from ..metrics.scores import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    auc_score = auc(fpr, tpr)

    print(f"ROC AUC Score: {auc_score:.4f}")

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color=color, linewidth=2.2, label=f"ROC (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", alpha=0.6, label="Random Chance (AUC = 0.5)")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.title(title)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if show:
        plt.show()


def residual_plot(y_true, y_pred, color="royalblue", title="Residual Plot", show=True):
    """Plot residuals (y_true - y_pred) against fitted predictions."""
    y_t = _to_numpy(y_true).ravel()
    y_p = _to_numpy(y_pred).ravel()
    residuals = y_t - y_p

    plt.figure(figsize=(7, 5))
    plt.scatter(y_p, residuals, color=color, alpha=0.75, edgecolors="k", linewidth=0.5)
    plt.axhline(0, color="crimson", linestyle="--", linewidth=1.8, label="Zero Error Line")

    plt.title(title)
    plt.xlabel("Fitted Predictions (y_pred)")
    plt.ylabel("Residuals (y_true - y_pred)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


def feature_importance(model, feature_names=None, top_n=10, color="royalblue", title="Feature Importance", show=True):
    """Plot ranked feature importances for Trees, Forests, Ensembles, or Linear models."""
    importances = None

    if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "w") and model.w is not None:
        w_abs = np.abs(np.asarray(model.w))
        sum_w = np.sum(w_abs)
        importances = w_abs / sum_w if sum_w > 0 else w_abs
    elif hasattr(model, "weights") and model.weights is not None:
        w_abs = np.abs(np.asarray(model.weights)[1:])
        sum_w = np.sum(w_abs)
        importances = w_abs / sum_w if sum_w > 0 else w_abs

    if importances is None or len(importances) == 0:
        print("[mymllib.feature_importance] Warning: Model does not have accessible feature importances or weights.")
        return

    n_feats = len(importances)
    if feature_names is not None:
        names = list(feature_names)
        if len(names) != n_feats:
            names = [str(n) for n in names[:n_feats]] + [f"Feature {i+1}" for i in range(len(names), n_feats)]
    else:
        names = [f"Feature {i+1}" for i in range(n_feats)]

    # Sort descending
    sorted_idx = np.argsort(importances)[::-1]
    top_k = min(int(top_n), n_feats)
    top_indices = sorted_idx[:top_k]

    top_names = [names[i] for i in top_indices][::-1]
    top_vals = [importances[i] for i in top_indices][::-1]

    plt.figure(figsize=(8, max(4, int(top_k * 0.45))))
    bars = plt.barh(range(top_k), top_vals, color=color, edgecolor="black", alpha=0.85)
    plt.yticks(range(top_k), top_names)
    plt.xlabel("Relative Importance Score")
    plt.title(title)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Annotate bar values
    for bar, val in zip(bars, top_vals):
        plt.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()
    if show:
        plt.show()

