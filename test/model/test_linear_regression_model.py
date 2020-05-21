import numpy as np
from unittest.mock import Mock
import pytest


from shfl.model.linear_regression_model import LinearRegressionModel


def test_linear_regression_model_initialize_single_target():
    n_features = 9
    n_targets = 1
    lnr = LinearRegressionModel(n_features = n_features, n_targets = n_targets)
    
    assert np.shape(lnr._model.intercept_) == ()
    assert np.shape(lnr._model.coef_) == (n_features,)
    assert np.shape(lnr.get_model_params()) == (n_targets, n_features + 1)


def test_linear_regression_model_initialize_multiple_targets():
    n_features = 9
    n_targets = 2
    lnr = LinearRegressionModel(n_features = n_features, n_targets = n_targets)
    
    assert np.shape(lnr._model.intercept_) == (n_targets,)
    assert np.shape(lnr._model.coef_) == (n_targets, n_features)
    assert np.shape(lnr.get_model_params()) == (n_targets, n_features + 1)