import random
import numpy as np
import tensorflow as tf


class Reproducibility:
    """
    Singleton class for ensure reproducibility
    You indicates the seed and the execution is the same

    Attributes:
        seed: Seed for the execution
        __instance: Singleton instance
    """
    __instance = None

    @staticmethod
    def getInstance():
        """
        Static access method.

        Return:
            Singleton instance class
        """
        if Reproducibility.__instance == None:
            Reproducibility()
        return Reproducibility.__instance

    def __init__(self, seed=None):
        """
        Virtually private constructor.
        """
        if Reproducibility.__instance != None:
             raise Exception("This class is a singleton")
        else:
            self.__seed = seed
            self.__seeds = {'server': self.__seed}
            Reproducibility.__instance = self

            if self.__seed is not None:
                self.set_seed('server')

    def set_seed(self, id):
        """
        Set server and clients seed

        Attributes:
            id: 'server' in server node and ID in client node
        """
        if not id in self.__seeds.keys():
            self.__seeds[id] = np.random.randint(2**32-1)
        np.random.seed(self.__seeds[id])
        random.seed(self.__seeds[id])
        tf.random.set_seed(self.__seeds[id])

