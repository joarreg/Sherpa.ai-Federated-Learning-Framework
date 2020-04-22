from shfl.private import Reproducibility
import numpy as np
import pytest


def test_reproducibility():
    seed = 1234
    Reproducibility(seed)

    assert Reproducibility.getInstance().seed == seed
    assert Reproducibility.getInstance().seeds['server'] == seed
    assert np.random.get_state()[1][0] == seed


def test_reproducibiliry_singleton():

    with pytest.raises(Exception):
        Reproducibility()


def test_set_seed():

    id = 'ID0'
    Reproducibility.getInstance().set_seed(id)

    assert Reproducibility.getInstance().seeds[id]
