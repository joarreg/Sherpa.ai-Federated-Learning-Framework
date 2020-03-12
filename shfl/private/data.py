from shfl.private.query import IdentityFunction
from shfl.differential_privacy.dp_mechanism import UnrandomizedMechanism


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


class DataAccessDefinition:
    """
    Class to represent how private data can be accessed.

    Data access definition is represented by two objects, a query and a differential privacy mechanism. It's necessary
    to define at least one of them to create a DataAccessDefinition

    # Arguments:
        query: Query to apply over data (see: [Query](../Query))
        dp_mechanism: Differential privacy mechanism to apply \
        (see: [Differential Privacy](../../Differential privacy/Mechanisms)).
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
    This class implements access to data without restrictions, plain data will be returned.
    """
    def __init__(self):
        super().__init__(IdentityFunction())
