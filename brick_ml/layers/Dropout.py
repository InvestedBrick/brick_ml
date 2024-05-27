import numpy as np
class Dropout:
    def __init__(self,dropout_rate : float) -> None:
        self.dropout_rate = dropout_rate
        self.mask = None
    def forward(self,inputs: np.ndarray, training : bool = True ):
        if training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate,size=inputs.shape)
            return inputs * self.mask
        else:
            return inputs * (1 - self.dropout_rate)
    def backward(self,output_gradient : np.ndarray,learning_rate : float):#learning rate just for consistancy
        return output_gradient * self.mask  