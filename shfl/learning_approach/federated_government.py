from sklearn.metrics import accuracy_score

from shfl.learning_approach.learning_approach import LearningApproach


class FederatedGovernment(LearningApproach):
    """
    Class used to represent Federated Government.

    Attributes
    ----------
    federated_data: FederatedData
        Federated data involved in the process
    model_builder: function
        function that builds a TrainableModel
    aggregator: str
        federated aggregator function
    """
    def get_global_model_accuracy(self, data_test, label_test):
        prediction = self._model.predict(data_test)
        accuracy_global_model_test = accuracy_score(label_test, prediction)
        print("Global model test Accuracy Client : " + str(accuracy_global_model_test))

    def deploy_central_model(self):
        for data_node in self._federated_data:
            data_node.set_model_params(self._model.get_model_params())

    def get_clients_accuracy(self, data_test, label_test):
        for data_node in self._federated_data:
            # Predict local model in test
            prediction = data_node.predict(data_test)
            accuracy_test = accuracy_score(label_test, prediction)

            print("Test Accuracy Client " + str(data_node) + ": " + str(accuracy_test))

    def train_all_clients(self):
        """
        Initialize the models of each client and train them
        """
        for data_node in self._federated_data:
            data_node.train_model(self._federated_data.identifier)

    def aggregate_weights(self):
        """
        Calculate aggregated weights and update clients and server models
        """
        weights = []
        for data_node in self._federated_data:
            weights.append(data_node.query_model_params())

        aggregated_weights = self._aggregator.aggregate_weights(weights)

        # Update server weights
        self._model.set_model_params(aggregated_weights)

    def run_rounds(self, n, test_data, test_label):
        """
        Run one more round beggining in the actual state
        """
        for i in range(0, n):
            print("Accuracy round " + str(i))
            self.deploy_central_model()
            self.train_all_clients()
            self.get_clients_accuracy(test_data, test_label)
            self.aggregate_weights()
            self.get_global_model_accuracy(test_data, test_label)
            print("\n\n")
