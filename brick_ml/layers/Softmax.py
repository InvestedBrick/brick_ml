import numpy as np
class Softmax:
    def __init__():
        pass

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        tmp = np.exp(inputs)
        self.output = tmp / np.sum(tmp) 
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        n = np.size(self.output)
        tmp = np.tile(self.output,n)
        return np.dot(tmp * (np.identity(n) - np.transpose(tmp)),output_gradient)