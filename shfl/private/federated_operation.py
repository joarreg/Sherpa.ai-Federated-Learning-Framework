import abc
from shfl.private.node import DataNode


class FederatedData:
    """
    Class representing data across different data nodes.

    This object is iterable over different data nodes. Every identifier for FederatedData \
    objects only can be used once.

    # Arguments:
        identifier : Unique identifier for the federated data
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
        """
        This method adds a new node containing data to the federated data

        # Arguments:
            node: DataNode object. (see: [DataNode](../DataNode))
            data: Data to add to this node
        """
        node.set_private_data(self._identifier, data)
        self._data_nodes.append(node)

    def num_nodes(self):
        """
        # Returns:
            num_nodes : The number of nodes in this federated data.
        """
        return len(self._data_nodes)

    def configure_data_access(self, data_access_definition):
        """
        Creates the same policy to access data over all the data nodes

        # Arguments:
            data_access_definition: (see: [Data](../Data/#dataaccessdefinition))
        """
        for data_node in self._data_nodes:
            data_node.configure_private_data_access(self._identifier, data_access_definition)

    def query(self):
        """
        Queries over every node and returns the answer of every node in a list

        # Returns
           result: List containing responses for every node
        """
        answer = []
        for data_node in self._data_nodes:
            answer.append(data_node.query_private_data(self._identifier))

        return answer


class FederatedTransformation(abc.ABC):
    """
    Interface defining method to apply an operation over FederatedData
    """
    @abc.abstractmethod
    def apply(self, data):
        """
        This method receives data to be modified and performs the required modifications over it.

        # Arguments:
            data: The object that has to be modified
        """


def federate_array(identifier, array, num_data_nodes):
    """
    Creates FederatedData from a indexable array.

    The array will be divided using the first dimension.

    # Arguments:
        identifier: String for unique identifier that will be used for the FederatedData
        array : Indexable array with any number of dimensions
        num_data_nodes: Number of nodes to use

    # Returns
        federated_array: FederatedData with an array of size len(array)/num_data_nodes in every node.
    """
    split_size = len(array) / float(num_data_nodes)
    last = 0.0

    federated_array = FederatedData(identifier)
    while last < len(array):
        federated_array.add_data_node(DataNode(), array[int(last):int(last + split_size)])
        last = last + split_size

    return federated_array


def apply_federated_transformation(federated_data, federated_transformation):
    """
    Applies the federated transformation over this federated data.

    Original federated data will be modified.

    # Arguments:
        federated_data: FederatedData to use in the transformation
        federated_transformation: FederatedTransformation that will be applied over this data
    """
    for data_node in federated_data:
        data_node.apply_data_transformation(federated_data.identifier, federated_transformation)
