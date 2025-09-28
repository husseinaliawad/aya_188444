import numpy as np
from sklearn.datasets import make_classification, make_regression
from engine.gafs import GeneticFeatureSelector


def test_ga_classification_runs_small():
    X, y = make_classification(n_samples=120, n_features=12, n_informative=5, random_state=0)
    ga = GeneticFeatureSelector(population_size=12, generations=5, random_state=0)
    res = ga.fit(X, y, feature_names=[f"f{i}" for i in range(X.shape[1])], task='classification')
    assert res.n_features == X.shape[1]
    assert isinstance(res.best_mask, list) and len(res.best_mask) == X.shape[1]
    assert any(res.best_mask), "At least one feature should be selected"


def test_ga_regression_runs_small():
    X, y = make_regression(n_samples=120, n_features=10, n_informative=4, noise=0.1, random_state=0)
    ga = GeneticFeatureSelector(population_size=12, generations=5, random_state=1)
    res = ga.fit(X, y, feature_names=[f"f{i}" for i in range(X.shape[1])], task='regression')
    assert res.n_features == X.shape[1]
    assert isinstance(res.best_mask, list) and len(res.best_mask) == X.shape[1]
    assert any(res.best_mask)
