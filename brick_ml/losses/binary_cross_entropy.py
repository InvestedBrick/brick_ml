import numpy as np

class binary_cross_entropy:
    def __init__(self) -> None:
        pass
    def loss(self,y_true,y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    def loss_derivative(self,y_true,y_pred):
        return((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)