import numpy as np
import pytest

import shfl.core.data
from shfl.core.data import FederatedData, LabeledData
from shfl.core.query import Get
from shfl.core.node import DataNode


def test_federate_array():
    data_size = 10000
    num_clients = 1000
    array = np.random.rand(data_size)
    federated_array = shfl.core.data.federate_array("my_array1", array, num_clients)
    assert federated_array.num_nodes() == num_clients
    assert federated_array.identifier == "my_array1"


def test_federate_array_size_private_data():
    data_size = 10000
    num_clients = 10
    array = np.random.rand(data_size)
    federated_array = shfl.core.data.federate_array("my_array", array, num_clients)
    for data_node in federated_array:
        assert len(data_node.query_private_data(Get(), "my_array")) == data_size/num_clients

    assert federated_array[0].query_private_data(Get(), "my_array")[0] == array[0]


def test_federated_data():
    data_size = 10
    federated_data = FederatedData("my_federated_data")
    assert federated_data.num_nodes() == 0
    array = np.random.rand(data_size)
    federated_data.add_data_node(DataNode(), array)
    assert federated_data.num_nodes() == 1
    assert federated_data[0].query_private_data(Get(), "my_federated_data")[0] == array[0]


def test_federated_data_identifier():
    data_size = 10
    federated_data = FederatedData("my_federated_data")
    array = np.random.rand(data_size)
    federated_data.add_data_node(DataNode(), array)
    with pytest.raises(KeyError):
        federated_data[0].query_private_data(Get(), "bad_identifier_federated_data")


def test_federated_data_repeated_identifier():
    my_data = FederatedData("my_federated_data")
    with pytest.raises(ValueError):
        FederatedData("my_federated_data")


def test_labeled_data():
    data = np.random.rand(10)
    label = np.random.rand(1)
    labeled_data = LabeledData(data, label)
    for i in range(len(data)):
        assert labeled_data.data[i] == data[i]
    assert labeled_data.label == label
    new_data = np.random.rand(10)
    labeled_data.data = new_data
    for i in range(len(new_data)):
        assert labeled_data.data[i] == new_data[i]
    new_label = np.random.rand(1)
    labeled_data.label = new_label
    assert labeled_data.label == new_label

