import numpy as np
import pytest

import shfl.core.federated_operation
from shfl.core.node import DataNode
from shfl.core.query import Get
from shfl.core.federated_operation import FederatedTransformation, FederatedData


class TestTransformation(FederatedTransformation):
    def apply(self, data):
        data += 1


def test_federate_transformation():
    random_array = np.random.rand(30)
    federated_array = shfl.core.federated_operation.federate_array("my_federated_array", random_array, 30)
    shfl.core.federated_operation.apply_federated_transformation(federated_array, TestTransformation())
    index = 0
    for data_node in federated_array:
        assert data_node.query_private_data(Get(), "my_federated_array") == random_array[index] + 1
        index = index + 1


def test_query_federate_data():
    random_array = np.random.rand(30)
    federated_array = shfl.core.federated_operation.federate_array("my_federated_array", random_array, 30)
    answer = shfl.core.federated_operation.query_federated_data(federated_array, Get())
    for i in range(len(answer)):
        assert answer[i] == random_array[i]


def test_federate_array():
    data_size = 10000
    num_clients = 1000
    array = np.random.rand(data_size)
    federated_array = shfl.core.federated_operation.federate_array("my_array1", array, num_clients)
    assert federated_array.num_nodes() == num_clients
    assert federated_array.identifier == "my_array1"


def test_federate_array_size_private_data():
    data_size = 10000
    num_clients = 10
    array = np.random.rand(data_size)
    federated_array = shfl.core.federated_operation.federate_array("my_array", array, num_clients)
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

