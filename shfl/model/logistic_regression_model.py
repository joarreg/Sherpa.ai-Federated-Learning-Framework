from shfl.model.model import TrainableModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


class LogisticRegressionModel(TrainableModel):
    """
    This class offers support for scikit-learn logistic regression model. It implements [TrainableModel](../model/#trainablemodel-class)

    # Arguments:
        n_features: integer number of features (independent variables).
        classes: array of classes to predict. At least 2 classes must be provided.
        model_inputs: optional dictionary containing the [model input parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    """
    def __init__(self, n_features, classes, model_inputs=None):
        if model_inputs is None:
            model_inputs = {}
        self._check_initialization(n_features, classes)
        self._model = LogisticRegression(**model_inputs)
        self._n_features = n_features
        classes = np.sort(np.asarray(classes))
        self._model.classes_ = classes
        n_classes = len(classes)
        if n_classes == 2:
            n_classes = 1
        self.set_model_params(np.zeros((n_classes, n_features + 1)))
        
    def train(self, data, labels):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments
            data: Data, array-like of shape (n_samples, n_features)
            labels: Target classes, array-like of shape (n_samples,) 
        """
        self._check_data(data)
        self._check_labels(labels)
        
        self._model.fit(data, labels)

    def predict(self, data):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        Arguments:
            data: Data, array-like of shape (n_samples, n_features)
        """
        self._check_data(data)
        
        prediction = self._model.predict(data)
        
        return prediction
    
    def evaluate(self, data, labels):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)
        Metrics for evaluating model's performance.
        
        Arguments:
            data: Data, array-like of shape (n_samples, n_features)
            labels: Target classes, array-like of shape (n_samples,) 
        """
        self._check_data(data)
        self._check_labels(labels)
        
        prediction = self.predict(data)
        bas = metrics.balanced_accuracy_score(labels, prediction)
        cks = metrics.cohen_kappa_score(labels, prediction)
        
        return bas, cks
    
    def performance(self, data, labels):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)
        
        Arguments:
            data: Data, array-like of shape (n_samples, n_features)
            labels: Target classes, array-like of shape (n_samples,) 
        """
        self._check_data(data)
        self._check_labels(labels)
        
        prediction = self.predict(data)
        bas = metrics.balanced_accuracy_score(labels, prediction)
        
        return bas

    def get_model_params(self):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)
        """
        
        return np.column_stack((self._model.intercept_, self._model.coef_))

    def set_model_params(self, params):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)
        """
        self._model.intercept_ = params[:,0]
        self._model.coef_ = params[:, 1:]

    def _check_data(self, data):
        """
        Method that checks whether the data dimension is correct.
        """
        if data.ndim == 1:
            if self._n_features != 1:
                raise AssertionError("Data need to have the same number of features described by the model, " + str(self._n_features)
                                     + ". Current data have only 1 feature.")
        elif data.shape[1] != self._n_features:
            raise AssertionError("Data need to have the same number of features described by the model, " + str(self._n_features) +
                                 ". Current data has " + str(data.shape[1]) + " features.")

    def _check_labels(self, labels):
        """
        Method that checks whether the classes are correct. 
        The classes in client's data must be the same as the input ones.
        
        # Arguments:
            labels: array with classes
        """
        classes = np.unique(np.asarray(labels))
        if not np.array_equal(self._model.classes_, classes):
            raise AssertionError("Labels need to have the same classes described by the model, " + str(self._model.classes_)
                                 + ". Labels of this node are " + str(classes) + " .")
    
    @staticmethod
    def _check_initialization(n_features, classes):
        """
        Method that checks if model's initialization is correct. 
        The number of features must be an integer equal or greater to one, and there must be at least two classes.

        # Arguments:
            n_features: number of features
            classes: array of classes to predict
        """
        if not isinstance(n_features, int):
            raise AssertionError("n_features must be a positive integer number. Provided " + str(n_features) + " features.")
        if n_features < 0:
            raise AssertionError("It must verify that n_features > 0. Provided value " + str(n_features) + ".")
        if len(classes) < 2:
            raise AssertionError("It must verify that the number of classes > 1. Provided " + str(len(classes)) + " classes.")
        if len(np.unique(classes)) != len(classes):
            classes = list(classes)
            duplicated_classes = [i_class for i_class in classes if classes.count(i_class) > 1]
            raise AssertionError("No duplicated classes allowed. Class(es) duplicated: " + str(duplicated_classes) )
