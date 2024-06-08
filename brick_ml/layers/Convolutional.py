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
        self.inputs = inputs
        batch_size = inputs.shape[0]
        self.correlation = np.zeros((batch_size,) + self.output_shape)
        for b in range(batch_size):
            for i in range(self.n_kernels):
                for j in range(self.input_depth):
                    self.correlation[b,i] += signal.correlate2d(self.inputs[b,j],self.kernels[i,j],mode='valid')
                self.correlation[b,i] += self.biases[i]    
        if self.activation is not None:
            self.output = self.activation.function(self.correlation)
        else:
            self.output = self.correlation   
        return self.output           
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        if self.activation is not None:
            output_gradient *= self.activation.function_derivative(self.correlation)
        
        kernels_gradient = np.zeros(self.kernels.shape)
        inputs_gradient = np.zeros(self.inputs.shape)
        batch_size = output_gradient.shape[0]
        for b in range(batch_size):
            for i in range(self.n_kernels):
                for j in range(self.input_depth):
                    kernels_gradient[i,j] = signal.correlate2d(self.inputs[b,j],output_gradient[b,i],mode='valid')
                    inputs_gradient[b,j] += signal.convolve2d(output_gradient[b,i],self.kernels[i,j],mode='full')
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * np.sum(output_gradient,axis=(0,2,3)) / batch_size
        return inputs_gradient   
