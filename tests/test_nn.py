import numpy as np
import pytest
from mymllib.nn.layers import Dense
from mymllib.nn.network import Network
from mymllib.nn.optimizers import Adam, SGD, Momentum, RMSProp

def test_nn_classification():
    X = np.random.randn(20, 5)
    y = np.random.randint(0, 2, 20)
    
    layers = [
        Dense(5, 10, activation='relu'),
        Dense(10, 2, activation='softmax')
    ]
    net = Network(layers=layers)
    net.fit(X, y, optimizer=Adam(lr=0.01), loss='cross_entropy', epochs=5, batch_size=4)
    
    preds = net.predict(X)
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1})

def test_nn_regression():
    X = np.random.randn(20, 5)
    y = np.random.randn(20)
    
    layers = [
        Dense(5, 10, activation='relu'),
        Dense(10, 1, activation='linear')
    ]
    net = Network(layers=layers)
    net.fit(X, y, optimizer=SGD(lr=0.01), loss='mse', epochs=5, batch_size=4)
    
    preds = net.predict(X)
    assert preds.shape == (20,) or preds.shape == (20, 1)

def test_nn_edge_case_single_sample():
    # Single sample
    X = np.random.randn(1, 5)
    y = np.array([1])
    
    layers = [
        Dense(5, 10, activation='relu'),
        Dense(10, 2, activation='softmax')
    ]
    net = Network(layers=layers)
    net.fit(X, y, optimizer=RMSProp(), loss='cross_entropy', epochs=2, batch_size=1)
    
    preds = net.predict(X)
    assert preds.shape == (1,)

def test_nn_optimizers():
    layers = [Dense(2, 2, activation='linear')]
    net = Network(layers=layers)
    X = np.random.randn(10, 2)
    y = np.random.randn(10, 2)
    
    # Test Momentum
    net.fit(X, y, optimizer=Momentum(), loss='mse', epochs=1)
