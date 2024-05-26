import numpy as np
class Dense:
    def __init__(self,n_inputs : int,n_neurons : int ,activation = None) -> None:
        self.n_neurons = n_neurons
        self.activation = activation
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.random.randn(1,n_neurons)
    def forward(self,inputs : np.ndarray):
        self.inputs = inputs
        self.weighted_sum = np.dot(inputs,self.weights) + self.biases
        if self.activation is not None:
            self.output = self.activation.function(self.weighted_sum)
        else:
            self.output =  self.weighted_sum
        return self.output    
    def backward(self, output_gradient: np.ndarray, learning_rate: float):
        if self.activation is not None:
            output_gradient *= self.activation.function_derivative(self.weighted_sum)
        d_weights = np.dot(self.inputs.T, output_gradient)
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * output_gradient
        return np.dot(output_gradient, self.weights.T)
        

