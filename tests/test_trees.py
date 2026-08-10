import pytest
import numpy as np
import pandas as pd
from mymllib.trees.tree import Tree
from mymllib.trees.forest import Forest
from mymllib.trees.bag import Bag
from mymllib.trees.adaboost import AdaBoost
from mymllib.trees.gradient_boost import GradientBoost
from mymllib.trees.xgboost import XGBoost

def test_tree():
    X = np.random.rand(20, 5)
    y = np.random.choice([0, 1], size=20)
    
    # Classify
    clf = Tree(task="classify", max_depth=3)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (20,)
    assert set(preds).issubset({0, 1})
    
    probas = clf.predict_proba(X)
    assert probas.shape == (20, 2)
    
    # Regress
    y_reg = np.random.rand(20)
    reg = Tree(task="regress", max_depth=3)
    reg.fit(X, y_reg)
    preds = reg.predict(X)
    assert preds.shape == (20,)
    
    # Edge case: single class
    y_single = np.zeros(20)
    clf_single = Tree(task="classify")
    clf_single.fit(X, y_single)
    preds_single = clf_single.predict(X)
    assert np.all(preds_single == 0)

def test_forest():
    X = np.random.rand(20, 5)
    y = np.random.choice([0, 1], size=20)
    
    # Classify
    clf = Forest(n_trees=5, task="classify", max_depth=3)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (20,)
    
    # Regress
    y_reg = np.random.rand(20)
    reg = Forest(n_trees=5, task="regress", max_depth=3)
    reg.fit(X, y_reg)
    preds = reg.predict(X)
    assert preds.shape == (20,)
    
def test_bag():
    X = np.random.rand(20, 5)
    y = np.random.choice([0, 1], size=20)
    
    # Bag
    clf = Bag(n_models=3)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (20,)

def test_adaboost():
    X = np.random.rand(20, 5)
    y = np.random.choice([0, 1], size=20)
    
    clf = AdaBoost(n_rounds=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (20,)
    
def test_gradient_boost():
    X = np.random.rand(20, 5)
    y = np.random.choice([0, 1], size=20)
    
    clf = GradientBoost(n_rounds=5, task="classify")
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (20,)

    y_reg = np.random.rand(20)
    reg = GradientBoost(n_rounds=5, task="regress")
    reg.fit(X, y_reg)
    preds = reg.predict(X)
    assert preds.shape == (20,)

def test_xgboost():
    X = np.random.rand(20, 5)
    y = np.random.choice([0, 1], size=20)
    
    clf = XGBoost(n_rounds=5, task="classify")
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (20,)

    y_reg = np.random.rand(20)
    reg = XGBoost(n_rounds=5, task="regress")
    reg.fit(X, y_reg)
    preds = reg.predict(X)
    assert preds.shape == (20,)
