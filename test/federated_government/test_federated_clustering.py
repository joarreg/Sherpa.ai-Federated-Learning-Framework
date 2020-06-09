from shfl.federated_government.federated_clustering import FederatedClustering
from shfl.data_distribution.data_distribution_iid import IidDataDistribution
from shfl.data_distribution.data_distribution_non_iid import NonIidDataDistribution
from shfl.federated_aggregator.cluster_fedavg_aggregator import ClusterFedAvgAggregator
from shfl.model.kmeans_model import KMeansModel

import numpy as np


def test_FederatedClustering():
    cfg = FederatedClustering('IRIS', iid=True, num_nodes=3, percent=20)

    assert cfg._test_data is not None
    assert cfg._test_labels is not None
    assert cfg._num_clusters == 3
    assert cfg._num_features == 4
    assert isinstance(cfg._aggregator, ClusterFedAvgAggregator)
    assert isinstance(cfg._model, KMeansModel)
    assert cfg._federated_data is not None

def test_FederatedClustering_wrong_database():
    cfg = FederatedClustering('MNIST', iid=True, num_nodes=3, percent=20)

    assert cfg._test_data is None