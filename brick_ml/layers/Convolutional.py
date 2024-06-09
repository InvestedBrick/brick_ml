import numpy as np
from scipy import signal
class Convolutional:
    def __init__(self,input_shape,kernel_size : int,n_kernels : int,activation=None) -> None:
        self.input_depth,self.input_height,self.input_width = input_shape
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.activation = activation
        self.output_shape = (n_kernels,self.input_height - kernel_size + 1,self.input_width - kernel_size + 1)
        self.kernel_shape = (n_kernels,self.input_depth,kernel_size,kernel_size)
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.rand(n_kernels)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the layer.

        Args:
        - inputs (np.ndarray): Input data with shape (batch_size, input_depth, input_height, input_width).
        - training (bool): Flag indicating whether the model is in training mode.

        Returns:
        - np.ndarray: Output of the layer with shape (batch_size, n_kernels, output_height, output_width).
        """
        
        # Store the inputs for backpropagation
        self.inputs = inputs
        
        # Initialize the output array with zeros
        batch_size = inputs.shape[0]
        self.correlation = np.zeros((batch_size,) + self.output_shape)
        
        # Perform convolution and accumulate the result in the correlation array
        for b in range(batch_size):
            for i in range(self.n_kernels):
                for j in range(self.input_depth):
                    # Perform convolution using the correlation function from scipy
                    self.correlation[b,i] += signal.correlate2d(self.inputs[b,j],self.kernels[i,j],mode='valid')
                # Add the biases to the correlation
                self.correlation[b,i] += self.biases[i]    
        
        # Apply the activation function if it exists
        if self.activation is not None:
            self.output = self.activation.function(self.correlation)
        else:
            self.output = self.correlation   
        
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
        # Apply the derivative of the activation function if it exists
        if self.activation is not None:
            output_gradient *= self.activation.function_derivative(self.correlation)
        
        # Calculate the gradient of the loss with respect to the weights and biases
        kernels_gradient = np.zeros(self.kernels.shape)  # Gradient of the loss with respect to the weights
        inputs_gradient = np.zeros(self.inputs.shape)  # Gradient of the loss with respect to the input
        batch_size = output_gradient.shape[0]
        for b in range(batch_size):
            for i in range(self.n_kernels):
                for j in range(self.input_depth):
                    # Calculate the gradient of the loss with respect to the weights
                    kernels_gradient[i,j] = signal.correlate2d(self.inputs[b,j],output_gradient[b,i],mode='valid')
                    # Calculate the gradient of the loss with respect to the input
                    inputs_gradient[b,j] += signal.convolve2d(output_gradient[b,i],self.kernels[i,j],mode='full')
        # Update the weights and biases using gradient descent
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * np.sum(output_gradient,axis=(0,2,3)) / batch_size
        return inputs_gradient   

