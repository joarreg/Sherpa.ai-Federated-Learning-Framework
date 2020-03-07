import numpy as np
import emnist

from shfl.data_base import data_base as db


class Emnist(db.DataBase):
    """
    Implementation for load EMNIST data
    """
    def __init__(self):
        super(Emnist, self).__init__()
       
    def load_data(self):
        """
        Load data from emnist package

        # Returns
            all_data : train data, train labels, validation data, validation labels, test data and test labels
        """
        images, labels = emnist.extract_training_samples('digits')
        labels = np.eye(10)[labels]
        self._test_data, self._test_labels = emnist.extract_test_samples('digits')

        dim_test = len(self._test_labels)
        self._train_data, self._train_labels, self._validation_data, self._validation_labels \
            = db.extract_validation_samples(images, labels, dim_test)

        self.shuffle()
        
        return self.data
