import numpy as np
class Softmax:
    def __init__(self):
        pass

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the layer.
        """
        # Subtract the max for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass through the layer.
        """
        # Create uninitialized array
        self.dinputs = np.empty_like(output_gradient)
        
        # Enumerate outputs and gradients
        for index, (single_output, single_gradient) in enumerate(zip(self.output, output_gradient)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_gradient)
        
        return self.dinputs

