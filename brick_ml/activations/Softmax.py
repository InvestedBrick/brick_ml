import numpy as np
class Softmax:
    def __init__(self) -> None:
        pass
    def function(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    def function_derivative(self,x):
        s = self.function(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

