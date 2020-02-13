from shfl.core.node import DataNode


def federate_array(identifier, array, num_data_nodes):
    """
    Create ~FederatedData from a numpy array.

    The array will be divided using the first dimension.

    Parameters
    ----------
    identifier : str
        Unique identifier that will be used for the FederatedData
    array : numpy array
        Numpy array with any number of dimensions
    num_data_nodes: int
        Number of nodes to use

    Returns
    -------
    ~FederatedData
        FederatedData with an array of size len(array)/num_data_nodes in every node.
    """
    split_size = len(array) / float(num_data_nodes)
    last = 0.0

    federated_array = FederatedData(identifier)
    while last < len(array):
        federated_array.add_data_node(DataNode(), array[int(last):int(last + split_size)])
        last = last + split_size

    return federated_array


class FederatedData:
    """
    Class representing data across different data nodes.

    Every identifier for FederatedData objects only can be used once.

    Attributes
    ----------
    _data_nodes : list
        List containing data nodes that are part of the federated data
    _identifier : str
        Unique identifier for the federated data
    """

    __used_identifiers = set()

    def __init__(self, identifier):
        if identifier in FederatedData.__used_identifiers:
            raise ValueError("Identifier " + str(identifier) + "is already in use")
        self._data_nodes = []
        self._identifier = identifier
        FederatedData.__used_identifiers.add(identifier)

    def __del__(self):
        FederatedData.__used_identifiers.remove(self._identifier)

    def __getitem__(self, item):
        return self._data_nodes[item]

    def __iter__(self):
        return iter(self._data_nodes)

    @property
    def identifier(self):
        return self._identifier

    def add_data_node(self, node, data):
        node.set_private_data(self._identifier, data)
        self._data_nodes.append(node)

    def num_nodes(self):
        return len(self._data_nodes)


class LabeledData:
    """
        Class to represent labeled data

    Attributes
    ----------
    _data : object
        Object representing data for a sample
    _label : object
        Label for this sample
    """
    def __init__(self, data, label):
        self._data = data
        self._label = label

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label
