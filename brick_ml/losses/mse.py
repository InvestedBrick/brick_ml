import numpy as np
class mse:
    def __init__(self) -> None:
        pass
    def loss(self,y_true,y_pred):
        return np.mean(np.square(y_true - y_pred))
    def loss_derivative(self,y_true,y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)