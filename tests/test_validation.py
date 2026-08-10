import numpy as np
import pytest
from mymllib.validation.cv import KFold, StratifiedKFold, LeaveOneOut, ShuffleSplit, TimeSeriesSplit, cross_validate
from mymllib.validation.search import GridSearchCV, RandomSearchCV

class DummyModel:
    def __init__(self, param1=1):
        self.param1 = param1
    
    def fit(self, X, y):
        pass
        
    def predict(self, X):
        return np.zeros(len(X))

def test_kfold():
    X = np.random.randn(20, 2)
    kf = KFold(n_splits=5)
    splits = list(kf.split(X))
    assert len(splits) == 5
    for train, test in splits:
        assert len(train) == 16
        assert len(test) == 4

def test_stratified_kfold():
    X = np.random.randn(20, 2)
    y = np.array([0]*10 + [1]*10)
    skf = StratifiedKFold(n_splits=2)
    splits = list(skf.split(X, y))
    assert len(splits) == 2
    for train, test in splits:
        assert len(train) == 10
        assert len(test) == 10
        assert np.sum(y[train]) == 5
        assert np.sum(y[test]) == 5

def test_cross_validate():
    X = np.random.randn(20, 2)
    y = np.random.randint(0, 2, 20)
    model = DummyModel()
    
    scores = cross_validate(model, X, y, cv=3)
    assert len(scores) == 3

def test_grid_search():
    X = np.random.randn(20, 2)
    y = np.random.randint(0, 2, 20)
    model = DummyModel()
    param_grid = {'param1': [1, 2]}
    
    gs = GridSearchCV(model, param_grid, cv=2)
    gs.fit(X, y)
    
    assert gs.best_params_ is not None
    assert gs.best_model_ is not None
    
    preds = gs.predict(X)
    assert len(preds) == 20

def test_random_search():
    X = np.random.randn(20, 2)
    y = np.random.randint(0, 2, 20)
    model = DummyModel()
    param_distributions = {'param1': [1, 2, 3, 4, 5]}
    
    rs = RandomSearchCV(model, param_distributions, n_iter=2, cv=2)
    rs.fit(X, y)
    
    assert rs.best_params_ is not None
    assert rs.best_model_ is not None

def test_edge_case():
    # cv number of splits equals n_samples
    X = np.random.randn(3, 2)
    y = np.array([0, 1, 0])
    
    kf = KFold(n_splits=3)
    splits = list(kf.split(X))
    assert len(splits) == 3
