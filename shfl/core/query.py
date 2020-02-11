import abc


class Query(abc.ABC):
    """
    This class represents a query over data
    """

    @abc.abstractmethod
    def get(self, data):
        """

        Parameters
        ----------
        data: object
            Data to process

        Returns
        -------
        answer: object
            Result of apply query over data
        """


class Get(Query):

    def get(self, data):
        return data
