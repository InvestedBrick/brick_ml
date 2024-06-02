import numpy as np
class Dense:
    def __init__(self,n_inputs : int,n_neurons : int ,activation = None) -> None:
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation
        self.init_weights_and_biases()
        
        
    def init_weights_and_biases(self):
        self.weights = np.random.randn(self.n_inputs,self.n_neurons)
        self.biases = np.random.randn(1,self.n_neurons)
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the layer.

        Args:
        - inputs (np.ndarray): Input data.
        - training (bool): Flag indicating whether the model is in training mode.

        Returns:
        - np.ndarray: Output of the layer.
        """

        # Store the inputs for backpropagation
        self.inputs = inputs

        # Calculate the weighted sum of inputs and weights plus biases
        self.weighted_sum = np.dot(inputs, self.weights) + self.biases

        # Apply the activation function if it exists
        if self.activation is not None:
            self.output = self.activation.function(self.weighted_sum)
        else:
            self.output = self.weighted_sum

        # Return the output of the layer
        return self.output
    def backward(self, output_gradient: np.ndarray, learning_rate: float):
        """
        Backpropagates the output gradient to update the weights and biases.

        Args:
        - output_gradient (np.ndarray): Gradient of the loss with respect to the output of the layer.
        - learning_rate (float): Learning rate for updating the weights and biases.

        Returns:
        - np.ndarray: Gradient of the loss with respect to the input of the layer.
        """
        # Apply the derivative of the activation function if it exists
        if self.activation is not None:
            output_gradient *= self.activation.function_derivative(self.weighted_sum)
        
        # Calculate the gradient of the loss with respect to the weights
        d_weights = np.dot(self.inputs.T, output_gradient)
        
        # Update the weights and biases using gradient descent
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        # Return the gradient of the loss with respect to the input of the layer
        return np.dot(output_gradient, self.weights.T)
        