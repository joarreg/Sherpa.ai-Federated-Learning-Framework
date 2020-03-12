import sklearn.datasets
from shfl.data_base import data_base as db


class CaliforniaHousing(db.DataBase):
    """
    This database loads the \
    [California housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#sklearn.datasets.fetch_california_housing)
    from sklearn, mainly for regression tasks.
    """
    def load_data(self):
        all_data = sklearn.datasets.fetch_california_housing()
        data = all_data["data"]
        labels = all_data["target"]

        test_size = int(len(data) * 0.1)
        validation_size = int(len(data) * 0.1)
        train_data, train_labels, self._test_data, self._test_labels = db.extract_validation_samples(data, labels,
                                                                                                     test_size)
        self._train_data, self._train_labels, self._validation_data, self._validation_labels \
            = db.extract_validation_samples(train_data, train_labels, validation_size)

        self.shuffle()

        return self.data
