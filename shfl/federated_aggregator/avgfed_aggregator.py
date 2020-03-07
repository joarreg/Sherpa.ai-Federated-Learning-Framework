import numpy as np

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class AvgFedAggregator(FederatedAggregator):
    """
    Implementation of Average Federated Aggregator. It only uses a simple average of the parameters of all the models
    """

    def aggregate_weights(self, clients_params):
        clients_params_array = np.array(clients_params)

        num_clients = clients_params_array.shape[0]
        num_layers = clients_params_array.shape[1]
        clients_params_array = clients_params_array.reshape(num_clients, num_layers)

        aggregated_weights = np.array([np.mean(clients_params_array[:, layer], axis=0) for layer in range(num_layers)])

        return aggregated_weights
