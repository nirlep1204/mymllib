import numpy as np
import pandas as pd
import pytest
from mymllib.data.preprocess import Scaler, Normalizer, Encoder, LabelEncoder, split, xy

def test_scaler():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = Scaler()
    X_scaled = scaler.fit_transform(X)
    
    assert np.allclose(np.mean(X_scaled, axis=0), [0.0, 0.0])
    assert np.allclose(np.std(X_scaled, axis=0), [1.0, 1.0])
    
    X_inv = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_inv)

def test_normalizer():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    norm = Normalizer()
    X_norm = norm.fit_transform(X)
    
    assert np.allclose(np.min(X_norm, axis=0), [0.0, 0.0])
    assert np.allclose(np.max(X_norm, axis=0), [1.0, 1.0])
    
    X_inv = norm.inverse_transform(X_norm)
    assert np.allclose(X, X_inv)

def test_encoder():
    df = pd.DataFrame({'color': ['red', 'blue', 'red']})
    enc = Encoder()
    df_encoded = enc.fit_transform(df, 'color')
    
    assert 'color_red' in df_encoded.columns
    assert 'color_blue' in df_encoded.columns
    assert df_encoded.shape == (3, 2)
    assert df_encoded['color_red'].tolist() == [1, 0, 1]

def test_label_encoder():
    y = ['cat', 'dog', 'cat', 'bird']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    assert len(np.unique(y_enc)) == 3
    assert len(y_enc) == 4
    
    y_inv = le.inverse(y_enc)
    assert list(y_inv) == y

def test_split():
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, 100)
    
    X_tr, X_te, y_tr, y_te = split(X, y, test=0.2)
    assert len(X_tr) == 80
    assert len(X_te) == 20
    assert len(y_tr) == 80
    assert len(y_te) == 20

def test_xy():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'target': [0, 1]})
    X, y = xy(df, 'target')
    
    assert 'target' not in X.columns
    assert list(y) == [0, 1]

def test_edge_case():
    # Constant column scaling
    X = np.array([[2.0], [2.0], [2.0]])
    scaler = Scaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled, [[0.0], [0.0], [0.0]])
