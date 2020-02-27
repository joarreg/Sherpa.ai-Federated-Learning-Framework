import numpy as np
import abc


class DifferentialPrivacyMechanism(abc.ABC):
    """
    This is the interface that must be implemented for every differential privacy method
    """
    @abc.abstractmethod
    def randomize(self, data):
        """
        The method should add some noise to the original data and return the obtained value
        """


class UnrandomizedMechanism(DifferentialPrivacyMechanism):
    """
    This class doesn't implement randomization mechanism
    """
    def randomize(self, data):
        """
        Data is returned without modification and no differential privacy is applied
        """
        return data


class RandomizeBinaryProperty(DifferentialPrivacyMechanism):
    """
    This class uses simple mechanism to add noise for binary data.

    Attributes
    ----------
    _noise : float
        Probability to use a random response instead of true value
    _mean : float
        Mean of the random response
    """
    def __init__(self, noise=0.5, mean=0.5):
        self._noise = noise
        self._mean = mean

    def randomize(self, data):
        """
        Answers with the truth with probability p_head and with random value with probability 1 - p_head
        """
        if data != 0 and data != 1:
            raise ValueError("RandomizeBinaryProperty works with binary data, but input is not binary")

        coin_flip = np.random.rand()
        if coin_flip > self._noise:
            return int(data)

        # Second coin flip
        coin_flip = np.random.rand()
        if coin_flip > self._mean:
            return 0

        return 1
