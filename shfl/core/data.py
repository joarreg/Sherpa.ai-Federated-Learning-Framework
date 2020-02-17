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
