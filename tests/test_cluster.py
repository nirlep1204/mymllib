import pytest
import numpy as np
import pandas as pd
from mymllib.cluster.kmeans import KMeans
from mymllib.cluster.gmm import GMM
from mymllib.cluster.hierarchical import Hierarchical

def test_kmeans():
    X = np.random.rand(20, 3)
    model = KMeans(k=2, seed=42)
    model.fit(X)
    
    assert model.centers is not None
    assert model.centers.shape == (2, 3)
    
    preds = model.predict(X)
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1})
    
    assert model.cost() >= 0.0

def test_kmeans_edge_case():
    # k > n_samples, some clusters will be empty and reinitialized
    X = np.random.rand(2, 3)
    model = KMeans(k=3, seed=42)
    model.fit(X)
    assert model.centers.shape == (3, 3)

def test_gmm():
    X = np.random.rand(20, 3)
    model = GMM(k=2, seed=42)
    model.fit(X)
    
    assert model.means is not None
    assert model.means.shape == (2, 3)
    assert model.covariances.shape == (2, 3, 3)
    
    preds = model.predict(X)
    assert preds.shape == (20,)
    
    probs = model.predict_proba(X)
    assert probs.shape == (20, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

def test_gmm_single_point():
    X = np.random.rand(1, 3)
    model = GMM(k=1, seed=42)
    model.fit(X)
    preds = model.predict(X)
    assert preds[0] == 0

def test_hierarchical():
    X = np.random.rand(10, 2)
    model = Hierarchical(k=2, linkage='single')
    model.fit(X)
    
    assert model.labels is not None
    assert model.labels.shape == (10,)
    assert set(np.unique(model.labels)).issubset({0, 1})
    
    dendrogram = model.dendrogram_data()
    # Number of merges should be n_samples - k
    assert len(dendrogram) == 8
