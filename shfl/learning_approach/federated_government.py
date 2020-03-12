from shfl.learning_approach.learning_approach import LearningApproach


class FederatedGovernment(LearningApproach):
    """
    Class used to represent Federated Government.
    """
    def evaluate_global_model(self, data_test, label_test):
        evaluation = self._model.evaluate(data_test, label_test)
        print("Global model test performance : " + str(evaluation))

    def deploy_central_model(self):
        for data_node in self._federated_data:
            data_node.set_model_params(self._model.get_model_params())

    def evaluate_clients(self, data_test, label_test):
        for data_node in self._federated_data:
            # Predict local model in test
            evaluation = data_node.evaluate(data_test, label_test)

            print("Test performance client " + str(data_node) + ": " + str(evaluation))

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
        Run one more round beginning in the actual state
        """
        for i in range(0, n):
            print("Accuracy round " + str(i))
            self.deploy_central_model()
            self.train_all_clients()
            self.evaluate_clients(test_data, test_label)
            self.aggregate_weights()
            self.evaluate_global_model(test_data, test_label)
            print("\n\n")
