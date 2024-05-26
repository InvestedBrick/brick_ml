import numpy as np
class Tanh:
    def __init__(self) -> None:
        pass
    def function(self,x):
        return np.tanh(x)
    def function_derivative(self,x):
        return 1 - np.square(self.function(x)) 