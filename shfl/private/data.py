import abc


class LabeledData:
    """
    Class to represent labeled data

    # Arguments:
        data: Features representing a data sample
        label: Label for this sample
    """
    def __init__(self, data, label):
        self._data = data
        self._label = label

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label


class DataAccessDefinition(abc.ABC):
    """
    Interface to implement in order to define how private data can be accessed.
    """

    @abc.abstractmethod
    def apply(self, data):
        """
        Every implementation needs to implement this method defining how data will be returned.

        # Arguments:
            data: Raw data that are going to be accessed

        # Returns
            result_data: Result data, function of argument data
        """


class UnprotectedAccess(DataAccessDefinition):
    """
    This class implements access to data without restrictions, plain data will be returned.
    """
    def apply(self, data):
        return data
