import abc


class Query(abc.ABC):
    """
    This class represents a query over private data
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


class IdentityFunction(Query):

    def get(self, data):
        return data
