import numpy as np

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class FedAvgAggregator(FederatedAggregator):
    """
    Implementation of Average Federated Aggregator. It only uses a simple average of the parameters of all the models.

    It implements [Federated Aggregator](../Federated Aggregator)
    """

    def aggregate_weights(self, clients_params):
        """
        Implementation of abstract method of class [AggregateWeightsFunction](../Federate Aggregator/#federatedaggregator-class)
        Arguments:
            clients_params: list of all clients' model parameters. 
            For each client, the model's parameters are assumed to be stored in a multi-dimensional numeric array.  

        # Returns
            aggregated_weights: aggregator weights representing the global learning model

        # References
            [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
        """

        clients_params_array = np.array(clients_params)
        print(clients_params_array.shape)

        num_clients = clients_params_array.shape[0]
        num_layers = clients_params_array.shape[1]

        aggregated_weights = np.array([np.mean(clients_params_array[:, layer], axis=0) for layer in range(num_layers)])

        return aggregated_weights
