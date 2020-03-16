import numpy as np
import pytest

import shfl.private.federated_operation
from shfl.private.federated_operation import FederatedTransformation
from shfl.private.federated_operation import FederatedData
from shfl.private.data import UnprotectedAccess


class TestTransformation(FederatedTransformation):
    def apply(self, data):
        data += 1


def test_federate_transformation():
    random_array = np.random.rand(30)
    federated_array = shfl.private.federated_operation.federate_array(random_array, 30)
    federated_array.configure_data_access(UnprotectedAccess())
    shfl.private.federated_operation.apply_federated_transformation(federated_array, TestTransformation())
    index = 0
    for data_node in federated_array:
        assert data_node.query() == random_array[index] + 1
        index = index + 1


def test_query_federate_data():
    random_array = np.random.rand(30)
    federated_array = shfl.private.federated_operation.federate_array(random_array, 30)
    federated_array.configure_data_access(UnprotectedAccess())
    answer = federated_array.query()
    for i in range(len(answer)):
        assert answer[i] == random_array[i]


def test_federate_array():
    data_size = 10000
    num_clients = 1000
    array = np.random.rand(data_size)
    federated_array = shfl.private.federated_operation.federate_array(array, num_clients)
    assert federated_array.num_nodes() == num_clients


def test_federate_array_size_private_data():
    data_size = 10000
    num_clients = 10
    array = np.random.rand(data_size)
    federated_array = shfl.private.federated_operation.federate_array(array, num_clients)
    federated_array.configure_data_access(UnprotectedAccess())
    for data_node in federated_array:
        assert len(data_node.query()) == data_size/num_clients

    assert federated_array[0].query()[0] == array[0]


def test_federated_data():
    data_size = 10
    federated_data = FederatedData()
    assert federated_data.num_nodes() == 0
    array = np.random.rand(data_size)
    federated_data.add_data_node(array)
    federated_data.configure_data_access(UnprotectedAccess())
    assert federated_data.num_nodes() == 1
    assert federated_data[0].query()[0] == array[0]


def test_federated_data_identifier():
    data_size = 10
    federated_data = FederatedData()
    array = np.random.rand(data_size)
    federated_data.add_data_node(array)
    federated_data.configure_data_access(UnprotectedAccess())
    with pytest.raises(ValueError):
        federated_data[0].query("bad_identifier_federated_data")
