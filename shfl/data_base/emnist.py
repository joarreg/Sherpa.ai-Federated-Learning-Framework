from emnist import extract_training_samples, extract_test_samples

import shfl.data_base.data_base as db


class Emnist(db.DataBase):
    """
    Implementation for load EMNIST data

    Attributes
    ----------
    _train_data : numpy matrix
        Data train
    _train_labels : numpy array
        Label for each train element
    _test_data : numpy matrix
        Data test
    _test_labels : numpy array
        Label for each test element
    _validation_data : numpy matrix
        Data from validation
    _validation_labels : numpy array
        Label for each validation data element
    """
    def __init__(self):
        super(Emnist, self).__init__()
       
    def load_data(self):
        """
        Load data from emnist package

        Return
        ------
        all_data : list
            Set of train data, label train, validation data, label validation, test data and label test
        """
        images, labels = extract_training_samples('digits')
        self._test_data, self._test_labels = extract_test_samples('digits')

        dim_test = len(self._test_labels)
        self._train_data, self._train_labels, self._validation_data, self._validation_labels \
            = db.extract_validation_samples(images, labels, dim_test)

        self.shuffle()
        
        return self.data
