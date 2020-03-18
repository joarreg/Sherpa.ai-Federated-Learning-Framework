import numpy as np
import pytest

import shfl
from shfl.private import DataNode
from shfl.private.data import DataAccessDefinition
from shfl.differential_privacy.dp_mechanism import UnrandomizedMechanism
from shfl.differential_privacy.dp_mechanism import RandomizedResponseBinary
from shfl.differential_privacy.dp_mechanism import RandomizedResponseCoins
from shfl.differential_privacy.dp_mechanism import LaplaceMechanism
from shfl.differential_privacy.probability_distribution import NormalDistribution


def test_unrandomized_mechanism():
    data_size = 100
    array = np.random.rand(data_size)
    federated_array = shfl.private.federated_operation.federate_array(array, data_size)
    federated_array.configure_data_access(DataAccessDefinition(dp_mechanism=UnrandomizedMechanism()))
    for i in range(data_size):
        assert len(federated_array[i].query()) == 1
        assert federated_array[i].query() == array[i]


def test_randomize_binary_mechanism_coins():
    data_size = 100
    array = np.ones(data_size)
    federated_array = shfl.private.federated_operation.federate_array(array, data_size)

    federated_array.configure_data_access(DataAccessDefinition(dp_mechanism=RandomizedResponseCoins()))

    result = federated_array.query()
    differences = 0
    for i in range(data_size):
        if result[i] != array[i]:
            differences = differences + 1

    assert 0 < differences < data_size
    assert np.mean(result) < 1


def test_randomize_binary_mechanism_array_coins():
    array = np.ones(100)
    node_single = DataNode()
    node_single.set_private_data(name="array", data=array)

    node_single.configure_data_access("array", DataAccessDefinition(dp_mechanism=RandomizedResponseCoins()))

    result = node_single.query("array")
    differences = 0
    for i in range(100):
        if result[i] != array[i]:
            differences = differences + 1

    assert not np.isscalar(result)
    assert 0 < differences < 100
    assert np.mean(result) < 1


def test_randomize_binary_mechanism_array_almost_always_true_values_coins():
    array = np.ones(1000)
    node_single = DataNode()
    node_single.set_private_data(name="array", data=array)

    # Very low heads probability in the first attempt, mean should be near true value
    data_access_definition = DataAccessDefinition(dp_mechanism=RandomizedResponseCoins(prob_head_first=0.01,
                                                                                       prob_head_second=0.9))
    node_single.configure_data_access("array", data_access_definition)

    result = node_single.query("array")

    assert 1 - np.mean(result) < 0.05


def test_randomize_binary_mechanism_array_almost_always_random_values_coins():
    array = np.ones(1000)
    node_single = DataNode()
    node_single.set_private_data(name="array", data=array)

    # Very high heads probability in the first attempt, mean should be near prob_head_second
    data_access_definition = DataAccessDefinition(dp_mechanism=RandomizedResponseCoins(prob_head_first=0.99,
                                                                                       prob_head_second=0.1))
    node_single.configure_data_access("array", data_access_definition)

    result = node_single.query("array")

    assert np.abs(0.1 - np.mean(result)) < 0.05


def test_randomize_binary_mechanism_scalar_coins():
    scalar = 1
    node_single = DataNode()
    node_single.set_private_data(name="scalar", data=scalar)

    node_single.configure_data_access("scalar", DataAccessDefinition(dp_mechanism=RandomizedResponseCoins()))

    result = node_single.query(private_property="scalar")

    assert np.isscalar(result)
    assert result == 0 or result == 1


def test_randomize_binary_mechanism_no_binary_coins():
    data_size = 100
    array = np.random.rand(data_size)
    federated_array = shfl.private.federated_operation.federate_array(array, data_size)

    federated_array.configure_data_access(DataAccessDefinition(dp_mechanism=RandomizedResponseCoins()))

    with pytest.raises(ValueError):
        federated_array.query()


def test_randomize_binary_deterministic():
    array = np.array([0, 1])
    node_single = DataNode()
    node_single.set_private_data(name="A", data=array)
    dp_mechanism = RandomizedResponseBinary(f0=1, f1=1)
    data_access_definition = DataAccessDefinition(dp_mechanism=dp_mechanism)
    node_single.configure_data_access("A", data_access_definition)

    result = node_single.query(private_property="A")

    assert np.array_equal(array, result)


def test_randomize_binary_random():
    data_size = 100
    array = np.ones(data_size)
    node_single = DataNode()
    node_single.set_private_data(name="A", data=array)
    dp_mechanism = RandomizedResponseBinary(f0=0.5, f1=0.5)
    data_access_definition = DataAccessDefinition(dp_mechanism=dp_mechanism)
    node_single.configure_data_access("A", data_access_definition)

    result = node_single.query(private_property="A")

    differences = 0
    for i in range(data_size):
        if result[i] != array[i]:
            differences = differences + 1

    assert 0 < differences < data_size
    assert np.mean(result) < 1


def test_randomize_binary_random_scalar_1():
    scalar = 1
    node_single = DataNode()
    node_single.set_private_data(name="scalar", data=scalar)
    dp_mechanism = RandomizedResponseBinary(f0=0.5, f1=0.5)
    data_access_definition = DataAccessDefinition(dp_mechanism=dp_mechanism)
    node_single.configure_data_access("scalar", data_access_definition)

    result = node_single.query(private_property="scalar")

    assert np.isscalar(result)
    assert result == 0 or result == 1


def test_randomize_binary_random_scalar_0():
    scalar = 0
    node_single = DataNode()
    node_single.set_private_data(name="scalar", data=scalar)
    dp_mechanism = RandomizedResponseBinary(f0=0.5, f1=0.5)
    data_access_definition = DataAccessDefinition(dp_mechanism=dp_mechanism)
    node_single.configure_data_access("scalar", data_access_definition)

    result = node_single.query(private_property="scalar")

    assert np.isscalar(result)
    assert result == 0 or result == 1


def test_randomize_binary_mechanism_array_almost_always_true_values_ones():
    array = np.ones(1000)
    node_single = DataNode()
    node_single.set_private_data(name="array", data=array)

    # Prob of one given 1 very high, mean should be near 1
    data_access_definition = DataAccessDefinition(dp_mechanism=RandomizedResponseBinary(f0=0.5, f1=0.99))
    node_single.configure_data_access("array", data_access_definition)

    result = node_single.query("array")

    assert 1 - np.mean(result) < 0.05


def test_randomize_binary_mechanism_array_almost_always_true_values_zeros():
    array = np.zeros(1000)
    node_single = DataNode()
    node_single.set_private_data(name="array", data=array)

    # Prob of one given 1 very high, mean should be near 1
    data_access_definition = DataAccessDefinition(dp_mechanism=RandomizedResponseBinary(f0=0.99, f1=0.5))
    node_single.configure_data_access("array", data_access_definition)

    result = node_single.query("array")

    assert np.abs(0 - np.mean(result)) < 0.05


def test_randomize_binary_mechanism_array_almost_always_false_values():
    array = np.ones(1000)
    node_single = DataNode()
    node_single.set_private_data(name="array", data=array)

    # Prob of one given 1 very low, mean should be near 0
    data_access_definition = DataAccessDefinition(dp_mechanism=RandomizedResponseBinary(f0=0.5, f1=0.01))
    node_single.configure_data_access("array", data_access_definition)

    result = node_single.query("array")

    assert np.abs(0 - np.mean(result)) < 0.05


def test_randomize_binary_mechanism_array_almost_always_false_values_zeros():
    array = np.zeros(1000)
    node_single = DataNode()
    node_single.set_private_data(name="array", data=array)

    # Prob of one given 1 very low, mean should be near 0
    data_access_definition = DataAccessDefinition(dp_mechanism=RandomizedResponseBinary(f0=0.01, f1=0.5))
    node_single.configure_data_access("array", data_access_definition)

    result = node_single.query("array")

    assert np.abs(1 - np.mean(result)) < 0.05


def test_randomize_binary_mechanism_no_binary():
    array = np.random.rand(1000)
    federated_array = shfl.private.federated_operation.federate_array(array, 100)

    federated_array.configure_data_access(DataAccessDefinition(dp_mechanism=RandomizedResponseBinary(f0=0.5, f1=0.5)))

    with pytest.raises(ValueError):
        federated_array.query()


def test_randomize_binary_mechanism_no_binary_scalar():
    scalar = 0.1
    node_single = DataNode()
    node_single.set_private_data(name="scalar", data=scalar)
    dp_mechanism = RandomizedResponseBinary(f0=0.5, f1=0.5)
    data_access_definition = DataAccessDefinition(dp_mechanism=dp_mechanism)
    node_single.configure_data_access("scalar", data_access_definition)

    with pytest.raises(ValueError):
        node_single.query("scalar")


def test_laplace_mechanism():
    data_size = 1000
    array = NormalDistribution(175, 7).sample(data_size)
    federated_array = shfl.private.federated_operation.federate_array(array, data_size)

    federated_array.configure_data_access(DataAccessDefinition(dp_mechanism=LaplaceMechanism(40, 1)))
    result = federated_array.query()

    differences = 0
    for i in range(data_size):
        if result[i] != array[i]:
            differences = differences + 1

    assert differences == data_size
    assert np.mean(array) - np.mean(result) < 5


def test_laplace_scalar_mechanism():
    scalar = 175

    node = DataNode()
    node.set_private_data("scalar", scalar)
    node.configure_data_access("scalar", DataAccessDefinition(dp_mechanism=LaplaceMechanism(40, 1)))

    result = node.query("scalar")

    assert scalar != result
    assert np.abs(scalar - result) < 100
