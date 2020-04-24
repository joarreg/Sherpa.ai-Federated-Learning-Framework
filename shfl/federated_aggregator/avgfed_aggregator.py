import numpy as np

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class AvgFedAggregator(FederatedAggregator):
    """
    Implementation of Average Federated Aggregator. It only uses a simple average of the parameters of all the models.

    It implements [Federated Aggregator](../Federated Aggregator)
    """

    def aggregate_weights(self, clients_params):
        """
        Implementation of abstract method of class [AggregateWeightsFunction](../Federate Aggregator/#federatedaggregator-class)

        # Returns
            aggregated_weights: aggregator weights representing the global learning model

        # References
            [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
        """

        clients_params_array = np.array(clients_params)

        num_clients = clients_params_array.shape[0]
        num_layers = clients_params_array.shape[1]
        clients_params_array = clients_params_array.reshape(num_clients, num_layers)

        aggregated_weights = np.array([np.mean(clients_params_array[:, layer], axis=0) for layer in range(num_layers)])

        return aggregated_weights
