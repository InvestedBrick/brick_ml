import numpy as np
class Softmax:
    def __init__():
        pass

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the layer.

        Args:
        - inputs (np.ndarray): Input data.
        - training (bool): Flag indicating whether the model is in training mode.

        Returns:
        - np.ndarray: Output of the layer.
        """
        # Calculate the exponential of the inputs
        tmp = np.exp(inputs)
        # Calculate the softmax function by dividing the exponential by the sum of the exponential
        self.output = tmp / np.sum(tmp, axis=1, keepdims=True) 
        # Return the output of the layer
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backpropagates the output gradient to update the weights and biases.

        Args:
        - output_gradient (np.ndarray): Gradient of the loss with respect to the output of the layer.
        - learning_rate (float): Learning rate for updating the weights and biases.

        Returns:
        - np.ndarray: Gradient of the loss with respect to the input of the layer.
        """
        # Calculate the number of elements in the output
        n = np.size(self.output)
        # Tile the output array to create a matrix of shape (n,n)
        tmp = np.tile(self.output,n)
        # Calculate the gradient of the loss with respect to the input
        return np.dot(tmp * (np.identity(n) - np.transpose(tmp)),output_gradient)

