class ReLU:
    def __init__(self) -> None:
        pass
    def function(self,x):
        return max(0,x)
    def function_derivative(self,x):
        return self.function(x) / x