import abc


class FederatedTransformation(abc.ABC):

    @abc.abstractmethod
    def apply(self, labeled_data):
        pass


def apply_federated_transformation(federated_data, federated_transformation):
    for data_node in federated_data:
        data_node.apply_data_transformation(federated_data.identifier, federated_transformation)


def query_federated_data(federated_data, query, identifier):
    answer = []
    for data_node in federated_data:
        answer.append(data_node.query_private_data(query, identifier))

    return answer
