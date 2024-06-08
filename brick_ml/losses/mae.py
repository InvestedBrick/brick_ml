import numpy as np
class mae:
    def __init__(self) -> None:
        pass
    def loss(self,y_true,y_pred):
        return np.mean(np.abs(y_true - y_pred))
    def loss_derivative(self,y_true,y_pred):
        return np.sign(y_true - y_pred)