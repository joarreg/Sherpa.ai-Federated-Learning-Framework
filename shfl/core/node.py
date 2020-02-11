class DataNode:
    """
    This class represents an independent node
    """

    def __init__(self):
        self._private_data = {}
        self._model = None

    @property
    def private_data(self):
        """
        Allows to see data for this node, but you cannot retrieve data

        Returns
        ------
        private_data : object
            test data
        """
        print("Node private data, you can see the data for debug purposes but the data remains in the node")
        print(type(self._private_data))
        print(self._private_data)

    def set_private_data(self, name, data):
        self._private_data[name] = data

    def apply_data_transformation(self, private_property, federated_transformation):
        federated_transformation.apply(self._private_data[private_property])

    def query_private_data(self,  query, private_property):
        return query.get(self._private_data[private_property])

    def query_model_params(self, query):
        return query.get(self._model.get_model_params())

    @property
    def model(self):
        print("You can't get the model, you need to query the params to access")
        print(type(self._model))
        print(self._model)

    @model.setter
    def model(self, model):
        self._model = model

    def set_model_params(self, model_params):
        self._model.set_model_params(model_params)

    def train_model(self, training_data_key):
        labeled_data = self._private_data.get(training_data_key)
        self._model.train(labeled_data.data, labeled_data.label)

    def predict(self, data):
        return self._model.predict(data)
