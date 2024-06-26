import numpy as np
#Work in progress...
class Pooling:
    def __init__(self, pool_size: tuple,stride : int, pad : int= 0,pool_function = "max") -> None:
        self.pool_size = pool_size
        self.stride = stride
        self.pad = pad
        self.pool_function = pool_function

            
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the pooling layer.

        Args:
        - inputs (np.ndarray): Input data with shape (batch_size, input_depth, input_height, input_width).
        - training (bool): Flag indicating whether the model is in training mode.

        Returns:
        - np.ndarray: Output of the layer with shape (batch_size, input_depth, output_height, output_width).
        """
        # Calculate the output dimensions
        batch_size, input_depth, input_height, input_width = inputs.shape
        pool_height, pool_width = self.pool_size

        output_height = (input_height - pool_height + 2 * self.pad) // self.stride + 1
        output_width = (input_width - pool_width + 2 * self.pad) // self.stride + 1

        # Initialize the output array with zeros
        output = np.zeros((batch_size, input_depth, output_height, output_width))

        # Perform pooling operation and store the result in the output array
        for b in range(batch_size):
            for c in range(input_depth):
                for h in range(0, pool_height + 1,self.stride):
                    for w in range(0, pool_width + 1,self.stride):
                        h_start = h
                        h_end = h + pool_height
                        w_start = w
                        w_end = w + pool_width
                        if(self.pool_function == "max"):
                            # Calculate the maximum value in the pooling window
                            output[b,c,h // self.stride, w // self.stride] = np.max(inputs[b,c,h_start:h_end, w_start:w_end])
                        elif(self.pool_function == "avg"):
                            # Calculate the average value in the pooling window
                            output[b,c,h // self.stride, w // self.stride] = np.mean(inputs[b,c,h_start:h_end, w_start:w_end])
        
        # Store the inputs for backpropagation
        self.inputs = inputs
        
        # Return the output of the layer
        return output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backpropagates the output gradient to update the weights and biases.

        Args:
        - output_gradient (np.ndarray): Gradient of the loss with respect to the output of the layer.
        - learning_rate (float): Learning rate for updating the weights and biases.

        Returns:
        - np.ndarray: Gradient of the loss with respect to the input of the layer.
        """
        # Get the shape of the input
        batch_size, input_depth, input_height, input_width = self.inputs.shape
        # Get the pooling dimensions
        pool_height, pool_width = self.pool_size

        # Initialize the input gradient with zeros
        input_gradient = np.zeros_like(self.inputs)

        # Perform backpropagation for each pooling window
        for b in range(batch_size):
            for c in range(input_depth):
                for h in range(0, pool_height + 1,self.stride):
                    for w in range(0, pool_width + 1,self.stride):
                        h_start = h
                        h_end = h + pool_height
                        w_start = w
                        w_end = w + pool_width
                        if(self.pool_function == "max"):
                            # Find the maximum value in the pooling window
                            max_val = np.max(self.inputs[b,c,h_start:h_end, w_start:w_end])
                            # Propagate the output gradient to the maximum value
                            for i in range(h_start,h_end):
                                for j in range(w_start,w_end):
                                    if self.inputs[b,c,i,j] == max_val:
                                        input_gradient[b,c,i,j] += output_gradient[b,c,h // self.stride, w // self.stride]
                        elif self.pool_function == "avg":
                            # Calculate the gradient for the average pooling
                            grad = output_gradient[b,c,h // self.stride, w // self.stride] / (pool_height * pool_width)
                            # Propagate the output gradient to each element in the pooling window
                            for i in range(h_start,h_end):
                                for j in range(w_start,w_end):
                                    input_gradient[b,c,i,j] += grad

        return input_gradient
