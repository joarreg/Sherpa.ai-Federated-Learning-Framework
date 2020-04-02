import abc
from shfl.private.federated_operation import FederatedTransformation, apply_federated_transformation
import random
import numpy as np


class FederatedDataAttack(abc.ABC, FederatedTransformation):
    """
    Interface defining method to apply an FederatedAttack over [FederatedData](./#federateddata-class)

    This interface implements [FederatedTransformation](./#federatedtransformation-class) in order to apply \
    data transformation to the node data.
    """

    @abc.abstractmethod
    def attack(self, federated_data):
        """
        This method receives federated data to be modified and performs the required modifications \
        (federated_attack) over it simulating the adversarial attack.

        # Arguments:
            federated_data: The data of nodes that have to be
        """


class FederatedPoisoningDataAttack(FederatedDataAttack):
    """
    Class representing poisoning data attack simulation. This simulation consists on shuffling \
    the labels of some nodes.

    This class implements interface [FederatedDataAttack](./#federateddataattack-class)

    # Arguments:
        percentage: percentage of nodes that are adversarial ones
    """

    def __init__(self, percentage):
        self._percentage = percentage

    def attack(self, federated_data):
        """
        Method that implements federated attack of data poisoning shuffling training labels of some nodes.
        """
        num_nodes = federated_data.num_nodes
        list_nodes = np.arange(num_nodes)
        list_adversaries = random.choice(list_nodes, k=int(self._percentage * num_nodes))
        federated_adversaries = federated_data[list_adversaries]

        # We apply federated transformation of data in selected nodes
        self.apply(federated_adversaries)

    def apply(self, labeled_data):
        """
        Method that implements apply abstract method of [FederatedTransformation](./#federatedtransformation-class) \
        shuffling labels of labeled_data
        """
        random.shuffle(labeled_data.label)