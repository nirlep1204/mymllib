import numpy as np
import pytest
from mymllib.metrics.scores import mse, rmse, mae, r2, accuracy, precision, recall, f1, confusion, roc_curve, auc, roc_auc_score, wcss, silhouette

def test_regression_metrics():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 3.0])
    
    assert mse(y, pred) == 0.0
    assert rmse(y, pred) == 0.0
    assert mae(y, pred) == 0.0
    assert r2(y, pred) == 1.0

    pred_wrong = np.array([2.0, 3.0, 4.0])
    assert mse(y, pred_wrong) == 1.0
    assert mae(y, pred_wrong) == 1.0

def test_classification_metrics():
    y = np.array([0, 1, 1, 0, 1])
    pred = np.array([0, 1, 0, 0, 1])
    
    acc = accuracy(y, pred)
    assert acc == 0.8
    
    p = precision(y, pred, average="binary")
    r = recall(y, pred, average="binary")
    f = f1(y, pred, average="binary")
    
    assert p > 0
    assert r > 0
    assert f > 0
    
    cm = confusion(y, pred)
    assert cm.shape == (2, 2)

def test_roc_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    assert len(fpr) > 0
    assert len(tpr) > 0
    assert len(thresholds) > 0
    
    score = roc_auc_score(y_true, y_scores)
    assert score > 0

def test_clustering_metrics():
    X = np.random.randn(20, 3)
    labels = np.random.randint(0, 3, 20)
    
    w = wcss(X, labels)
    s = silhouette(X, labels)
    
    assert w >= 0
    assert s >= -1 and s <= 1

def test_edge_case():
    # Only one class present in classification
    y = np.array([1, 1, 1])
    pred = np.array([1, 1, 1])
    assert accuracy(y, pred) == 1.0
    assert confusion(y, pred).shape == (1, 1)
