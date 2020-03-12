from keras.datasets import fashion_mnist

from shfl.data_base import data_base
from shfl.data_base.data_base import DataBase


class FashionMnist(DataBase):
    """
    Implementation for load FASHION-EMNIST data
    """
    def __init__(self):
        super(FashionMnist, self).__init__()

    def load_data(self):
        """
        Load data from emnist package

        # Returns:
            all_data : train data, train label, validation data, validation label, test data and test labels
        """
        ((images, labels), (self._test_data, self._test_labels)) = fashion_mnist.load_data()

        dim_test = len(self._test_labels)
        self._train_data, self._train_labels, self._validation_data, self._validation_labels \
            = data_base.extract_validation_samples(images, labels, dim_test)

        self.shuffle()
        
        return self._train_data, self._train_labels, self._validation_data, \
            self._validation_labels, self._test_data, self._test_labels
