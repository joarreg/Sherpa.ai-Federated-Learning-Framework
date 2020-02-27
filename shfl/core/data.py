from shfl.core.query import IdentityFunction
from shfl.core.dp_mechanism import UnrandomizedMechanism


class LabeledData:
    """
        Class to represent labeled data

    Attributes
    ----------
    _data : object
        Object representing data for a sample
    _label : object
        Label for this sample
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


class DataAccessDefinition:
    """
    Class to represent how private data can be accessed

    Attributes
    ----------
    _query : ~Query
        Function to apply to data before return
    _dp_mechanism : ~DifferentialPrivacyMechanism
        Randomization algorithm to apply after query
    """
    def __init__(self, query=None, dp_mechanism=None):
        if query is None and dp_mechanism is None:
            raise ValueError("You can't define a data access without setting one of query or dp_mechanism")

        if query is None:
            self._query = IdentityFunction()
        else:
            self._query = query

        if dp_mechanism is None:
            self._dp_mechanism = UnrandomizedMechanism()
        else:
            self._dp_mechanism = dp_mechanism

    @property
    def query(self):
        return self._query

    @property
    def dp_mechanism(self):
        return self._dp_mechanism


class UnprotectedAccess(DataAccessDefinition):
    """
    This class implements access to data without security
    """
    def __init__(self):
        super().__init__(IdentityFunction())
