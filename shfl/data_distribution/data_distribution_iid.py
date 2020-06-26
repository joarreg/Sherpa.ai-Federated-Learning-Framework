import numpy as np

from shfl.data_distribution.data_distribution import DataDistribution


class IidDataDistribution(DataDistribution):
    """
    Implementation of an independent and identically distributed data distribution using \
        [Data Distribution](../data_distribution/#datadistribution-class)
    """

    def make_data_federated(self, data, labels, num_nodes, percent, weights, sampling="without_replacement"):
        """
        Method that makes data and labels argument federated in an iid scenario.

        # Arguments:
            data: Data to federate
            labels: Labels to federate
            num_nodes: Number of nodes to create
            percent: Percent of the data (between 0 and 100) to be distributed
            weights: Array of weights for weighted distribution (default is None)
            sampling: methodology between with or without sampling (default "without_sampling")

        # Returns:
              * **federated_data, federated_labels**

        """
        if weights is None:
            weights = np.full(num_nodes, 1/num_nodes)

        # Shuffle data
        randomize = np.arange(len(labels))
        np.random.shuffle(randomize)
        data = data[randomize, ]
        labels = labels[randomize]

        # Select percent
        data = data[0:int(percent * len(data) / 100), ]
        labels = labels[0:int(percent * len(labels) / 100)]

        federated_data = []
        federated_label = []

        if sampling == "without_replacement":
            if sum(weights) > 1:
                weights = np.array([float(i)/sum(weights) for i in weights])

            sum_used = 0
            percentage_used = 0

            for client in range(0, num_nodes):
                federated_data.append(np.array(data[sum_used:int((percentage_used + weights[client]) * len(data)), ]))
                federated_label.append(np.array(labels[sum_used:int((percentage_used + weights[client]) * len(labels))]))

                sum_used = int((percentage_used + weights[client]) * len(data))
                percentage_used += weights[client]
        else:
            randomize = np.arange(len(labels))
            for client in range(0, num_nodes):
                federated_data.append(np.array(data[:int((weights[client]) * len(data)), ]))
                federated_label.append(np.array(labels[:int((weights[client]) * len(labels))]))

                np.random.shuffle(randomize)
                data = data[randomize, ]
                labels = labels[randomize]

        federated_data = np.array(federated_data)
        federated_label = np.array(federated_label)

        return federated_data, federated_label
