import pytest
import numpy as np
import pandas as pd
from mymllib.reduce.factor import Factor
from mymllib.reduce.ica import ICA
from mymllib.reduce.pca import PCA

def test_pca():
    X = np.random.rand(20, 5)
    pca = PCA(n_components=2)
    pca.fit(X)
    X_trans = pca.transform(X)
    assert X_trans.shape == (20, 2)
    
    X_rec = pca.reconstruct(X_trans)
    assert X_rec.shape == (20, 5)
    
    X_small = np.random.rand(3, 5)
    pca_small = PCA(n_components=3)
    X_small_trans = pca_small.fit_transform(X_small)
    assert X_small_trans.shape == (3, 3)

def test_ica():
    X = np.random.rand(20, 5)
    ica = ICA(n_components=2, max_iter=5)
    ica.fit(X)
    X_trans = ica.transform(X)
    assert X_trans.shape == (20, 2)
    
    X_trans2 = ica.fit_transform(X)
    assert X_trans2.shape == (20, 2)

def test_factor():
    X = np.random.rand(20, 5)
    factor = Factor(n_factors=2, max_iter=5)
    factor.fit(X)
    X_trans = factor.transform(X)
    assert X_trans.shape == (20, 2)
    
    X_trans2 = factor.fit_transform(X)
    assert X_trans2.shape == (20, 2)
