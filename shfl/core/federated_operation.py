import abc


class FederatedTransformation(abc.ABC):
    """
    Interface defining method to apply an operation over FederatedData
    """
    @abc.abstractmethod
    def apply(self, data):
        pass


def apply_federated_transformation(federated_data, federated_transformation):
    """
    Applies the federated transformation over this federated data

    Original federated data will be modified.

    Parameters
    ----------
    federated_data : ~FederatedData
        ~FederatedData to use in the transformation
    federated_transformation : ~FederatedTransformation
        ~FederatedTransformation that will be applied over this data

    """
    for data_node in federated_data:
        data_node.apply_data_transformation(federated_data.identifier, federated_transformation)


def query_federated_data(federated_data, query):
    """
    Apply the federated query over every node and returns the answer of every node in a list

    Parameters
    ----------
    federated_data : ~FederatedData
        ~FederatedData to use in the query
    query : ~Query
        ~Query that will be applied over this data

    Returns
    -------
    list
       List containing responses for every node
    """
    answer = []
    for data_node in federated_data:
        answer.append(data_node.query_private_data(query, federated_data.identifier))

    return answer
