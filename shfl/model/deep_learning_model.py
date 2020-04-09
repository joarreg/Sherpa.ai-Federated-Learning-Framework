from keras.callbacks import EarlyStopping
from shfl.model.model import TrainableModel
import tensorflow as tf


class DeepLearningModel(TrainableModel):
    """
    This class offers support for Keras and tensorflow models.

    # Arguments:
        model: compiled model, ready to train
        batch_size: batch_size to apply
        epochs: number of epochs
        initialized: Indicates if the model is initialized or not (default False)
    """
    def __init__(self, model, batch_size=None, epochs=1, initialized=False):
        if not initialized:
            self._model = model
        else:
            self._model = tf.keras.models.model_from_config({'class_name': model.__class__.__name__,
                                                             'config': model.get_config()})
            self._model.compile(optimizer=model.optimizer.__class__.__name__, loss=model.loss,
                                metrics=model.metrics_names[1:])

            self._model.set_weights(model.get_weights())

        self._data_shape = model.layers[0].get_input_shape_at(0)[1:]
        self._labels_shape = model.layers[-1].get_output_shape_at(0)[1:]

        self._batch_size = batch_size
        self._epochs = epochs

    def train(self, data, labels):
        """
        Method for train the model

        Arguments:
            data: matrix
                Data with shape NxD (N: Number of elements; D: Dimensions)
            labels: int matrix
                Labels for data with One Hot Encoded format.
        """
        self._check_data(data)
        self._check_labels(labels)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
        self._model.fit(x=data, y=labels, batch_size=self._batch_size, epochs=self._epochs, validation_split=0.2,
                        verbose=0, shuffle=False, callbacks=[early_stopping])

    def predict(self, data):
        """
        Method for train the model

        Arguments:
            data: matrix
                Data with shape NxD (N: Number of elements; D: Dimensions)
        """
        self._check_data(data)

        return self._model.predict(data, batch_size=self._batch_size).argmax(axis=-1)

    def evaluate(self, data, labels):
        """
        Method for evaluate the model

        Arguments:
            data: matrix
                Data with shape NxD (N: Number of elements; D: Dimensions)
            labels: int matrix
                Labels for data with One Hot Encoded format.
        """
        self._check_data(data)
        self._check_labels(labels)

        return self._model.evaluate(data, labels, verbose=0)

    def get_model_params(self):
        return self._model.get_weights()

    def set_model_params(self, params):
        self._model.set_weights(params)

    def _check_data(self, data):
        if data.shape[1:] != self._data_shape:
            raise AssertionError("Data need to have the same shape described by the model " + str(self._data_shape) +
                                 " .Current data has shape " + str(data.shape[1:]))

    def _check_labels(self, labels):
        if labels.shape[1:] != self._labels_shape:
            raise AssertionError("Labels need to have the same shape described by the model " + str(self._labels_shape)
                                 + " .Current data has shape " + str(labels.shape[1:]))
