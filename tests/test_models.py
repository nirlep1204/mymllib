import pytest
import numpy as np
import pandas as pd

from mymllib.models.gda import GDA
from mymllib.models.glm import GLM
from mymllib.models.knn import KNN
from mymllib.models.linear import Linear
from mymllib.models.logistic import Logistic
from mymllib.models.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from mymllib.models.perceptron import Perceptron
from mymllib.models.softmax import Softmax
from mymllib.models.svm import SVM
from mymllib.models.svr import SVR


def test_gda():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    model = GDA(shared_cov=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1})
    
    probs = model.predict_proba(X)
    assert probs.shape == (20, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_gda_edge_case():
    # test unshared cov
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    model = GDA(shared_cov=False)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)


def test_glm():
    X = np.random.rand(20, 3)
    y = np.random.rand(20)
    model = GLM(family='gaussian', max_iter=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)


def test_glm_edge_case():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    model = GLM(family='bernoulli', max_iter=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_knn():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    model = KNN(k=3, task='classify')
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_knn_edge_case():
    # k > n_samples
    X = np.random.rand(5, 3)
    y = np.random.rand(5)
    model = KNN(k=10, task='regress')
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (5,)


def test_linear():
    X = np.random.rand(20, 3)
    y = np.random.rand(20)
    model = Linear(method='gd', max_iter=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)


def test_linear_edge_case():
    X = np.random.rand(20, 3)
    y = np.random.rand(20)
    model = Linear(method='normal')
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)
    cost = model.cost(X, y)
    assert cost >= 0.0


def test_logistic():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    model = Logistic(max_iter=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_logistic_edge_case():
    # test reg
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    model = Logistic(max_iter=10, reg='l2', alpha=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)


def test_naive_bayes():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    model = GaussianNB()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)


def test_naive_bayes_edge_case():
    X = np.random.randint(0, 5, size=(20, 3))
    y = np.random.randint(0, 2, size=20)
    model = MultinomialNB()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)
    
    X2 = np.random.randint(0, 2, size=(20, 3))
    model2 = BernoulliNB()
    model2.fit(X2, y)
    preds2 = model2.predict(X2)
    assert preds2.shape == (20,)


def test_perceptron():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    model = Perceptron(max_iter=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_perceptron_edge_case():
    # Test completely separable data
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    model = Perceptron(max_iter=100)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_softmax():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 3, size=20)
    model = Softmax(max_iter=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_softmax_edge_case():
    # test single class
    X = np.random.rand(20, 3)
    y = np.zeros(20)
    model = Softmax(max_iter=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)) == {0}


def test_svm():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    model = SVM(kernel='linear', max_iter=5)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_svm_edge_case():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    model = SVM(kernel='rbf', max_iter=5)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)


def test_svr():
    X = np.random.rand(20, 3)
    y = np.random.rand(20)
    model = SVR(kernel='linear', max_iter=5)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)


def test_svr_edge_case():
    X = np.random.rand(20, 3)
    y = np.random.rand(20)
    model = SVR(kernel='rbf', max_iter=5)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)
