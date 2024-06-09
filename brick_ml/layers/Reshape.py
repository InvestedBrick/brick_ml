import numpy as np

class Reshape:
    def __init__(self, input_shape,output_shape,add_batch_size = True) -> None:
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.add_batch_size = add_batch_size
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        batch_size = inputs.shape[0]
        return np.reshape(inputs,(batch_size,) + self.output_shape)  
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        batch_size = output_gradient.shape[0]
        if self.add_batch_size: 
            return np.reshape(output_gradient,(batch_size,) + self.input_shape)
        else: 
            return np.reshape(output_gradient,self.input_shape)