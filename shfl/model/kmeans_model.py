from shfl.model.model import TrainableModel
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics


class KMeansModel(TrainableModel):
    """
    This class offers support for scikit-learn K-Means model. It implements [TrainableModel](../Model/#trainablemodel-class)

    # Arguments:
        n_clusters: number of clusters.
        init: Method of initialization. {‘k-means++’, ‘random’, ndarray}, default=’k-means++’.
            If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
            When ‘random’: choose n_clusters observations  (rows) at random from data for the initial centroids.
        n_init: Number of time the k-means algorithm will be run with different centroid seeds (default 10).
    """

    def __init__(self, n_clusters, init, n_init=10):
        self._k_means = KMeans(n_clusters=n_clusters, init=init, n_init=n_init)
        self._k_means.cluster_centers_ = init

    def train(self, data, labels=None):
        """
        Implementation of abstract method of class [TrainableModel](../Model/#trainablemodel-class)

        # Arguments
            data: Data, array-like of shape (n_samples, n_features)
            labels: None.
        """
        self._k_means.fit(data)

    def predict(self, data):
        """
        Implementation of abstract method of class [TrainableModel](../Model/#trainablemodel-class)

        Arguments:
            data: Data, array-like of shape (n_samples, n_features)
        """
        predicted_labels = self._k_means.predict(data)
        return predicted_labels

    def evaluate(self, data, labels):
        """
        Implementation of abstract method of class [TrainableModel](../Model/#trainablemodel-class)
        Metrics for evaluating model's performance.

        Arguments:
            data: Data, array-like of shape (n_samples, n_features)
            labels: Target, array-like of shape (n_samples,) or (n_samples, n_targets)
        """
        prediction = self.predict(data)

        homo = metrics.homogeneity_score(labels, prediction)
        compl = metrics.completeness_score(labels, prediction)
        v_meas = metrics.v_measure_score(labels, prediction)
        rai = metrics.adjusted_rand_score(labels, prediction)
        return homo, compl, v_meas, rai

    def get_model_params(self):
        """
        Implementation of abstract method of class [TrainableModel](../Model/#trainablemodel-class)
        """
        return self._k_means.cluster_centers_

    def set_model_params(self, params):
        """
        Implementation of abstract method of class [TrainableModel](../Model/#trainablemodel-class)
        """
        n_clusters = params.shape[0]
        self.__init__(n_clusters=n_clusters, init=params)