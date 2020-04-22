import abc


class LearningApproach(abc.ABC):
    """
    Abstract class Class used to represent a Learning Approach.

    # Arguments:
        federated_data: Federated data to use. (see: [FederatedData](../../private/federated_operation/#federateddata-class))
        model_builder: Function that return a trainable model (see: [Model](../../model))
        aggregator: federated aggregator function (see: [Federated Aggregator](../../federated_aggregator))
    """
    def __init__(self, model_builder, federated_data, aggregator):
        self._federated_data = federated_data
        self._model = model_builder()
        self._aggregator = aggregator
        for data_node in federated_data:
            data_node.model = model_builder()

    @abc.abstractmethod
    def train_all_clients(self):
        """
        Initialize the models of each client and train them
        """

    @abc.abstractmethod
    def aggregate_weights(self):
        """
        Calculate aggregated weights and update clients and server models
        """

    @abc.abstractmethod
    def run_rounds(self, n, test_data, test_label):
        """
        Run one more round beggining in the actual state
        """
