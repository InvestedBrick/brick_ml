import numpy as np
class ReLU:
    def __init__(self) -> None:
        pass
    def function(self,x):
        return np.maximum(0,x)
    def function_derivative(self,x):
        return np.where(x > 0, 1, 0)