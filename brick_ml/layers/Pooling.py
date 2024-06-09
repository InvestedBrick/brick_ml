import numpy as np
#Work in progress...
class Pooling:
    def __init__(self, pool_size: tuple,stride : int, pad : int= 0,pool_function = "max") -> None:
        self.pool_size = pool_size
        self.stride = stride
        self.pad = pad
        self.pool_function = pool_size

            
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        batch_size, input_depth, input_height, input_width = inputs.shape
        pool_height, pool_width = self.pool_size

        output_height = (input_height - pool_height + 2 * self.pad) // self.stride + 1
        output_width = (input_width - pool_width + 2 * self.pad) // self.stride + 1
        output = np.zeros((batch_size, input_depth, output_height, output_width))

        for b in range(batch_size):
            for c in range(input_depth):
                for h in range(0, pool_height + 1,self.stride):
                    for w in range(0, pool_width + 1,self.stride):
                        h_start = h
                        h_end = h + pool_height
                        w_start = w
                        w_end = w + pool_width
                        if(self.pool_function == "max"):
                            output[b,c,h // self.stride, w // self.stride] = np.max(inputs[b,c,h_start:h_end, w_start:w_end])
                        elif(self.pool_function == "avg"):
                            output[b,c,h // self.stride, w // self.stride] = np.mean(inputs[b,c,h_start:h_end, w_start:w_end])
        self.inputs = inputs
        return output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        batch_size, input_depth, input_height, input_width = self.inputs.shape
        pool_height, pool_width = self.pool_size

        input_gradient = np.zeros_like(self.inputs)

        for b in range(batch_size):
            for c in range(input_depth):
                for h in range(0, pool_height + 1,self.stride):
                    for w in range(0, pool_width + 1,self.stride):
                        h_start = h
                        h_end = h + pool_height
                        w_start = w
                        w_end = w + pool_width
                        if(self.pool_function == "max"):
                            max_val = np.max(self.inputs[b,c,h_start:h_end, w_start:w_end])
                            for i in range(h_start,h_end):
                                for j in range(w_start,w_end):
                                    if self.inputs[b,c,i,j] == max_val:
                                        input_gradient[b,c,i,j] += output_gradient[b,c,h // self.stride, w // self.stride]
                        elif self.pool_function == "avg":
                            grad = output_gradient[b,c,h // self.stride, w // self.stride] / (pool_height * pool_width)
                            for i in range(h_start,h_end):
                                for j in range(w_start,w_end):
                                    input_gradient[b,c,i,j] += grad

        return input_gradient                                