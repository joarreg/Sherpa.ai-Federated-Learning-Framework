import abc


class TrainableModel(abc.ABC):
    """
    Interface of the models that can be trained
    """

    @abc.abstractmethod
    def train(self, data, labels):
        """
        Method that train the model

        Parameters
        ----------
        data : numpy matrix
            Data to train the model
        labels: numpy matrix
            Label for each train element
        """

    @abc.abstractmethod
    def predict(self, data):
        """
        Predict labels for data

        Parameters
        ----------
        data: matrix
            data to classify

        Return
        ------
        predictions : matrix
            predictions for data
        """

    @abc.abstractmethod
    def get_model_params(self):
        """
        Gets the params that define the model
        Returns
        -------
        params : numpy array
            parameters defining the model
        """

    @abc.abstractmethod
    def set_model_params(self, params):
        """
        Update the params that define the model
        """
