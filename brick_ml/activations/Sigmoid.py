import numpy as np
class Sigmoid:
    def __init__(self) -> None:
        pass
    def function(self,x):
        return 1/(1+ np.exp(-x))
    def function_derivative(self,x):
        return self.function(x) * (1 - self.function(x))
        