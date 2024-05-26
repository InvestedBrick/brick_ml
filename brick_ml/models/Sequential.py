import numpy as np
class Sequential:
    def __init__(self,learning_rate : float, loss) -> None:
        self.layers = []
        self.loss_history = []
        self.learning_rate = learning_rate
        self.loss = loss
    def add_layer(self,layer):
        self.layers.append(layer)    
    def train(self, n_epochs: int, timestep: int, inputs: np.ndarray, expected_output: np.ndarray):
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for x,y in zip(inputs,expected_output):
                output = self.predict(x.reshape(1,-1))

                loss = self.loss.loss(y,output)
                epoch_loss += loss
                gradient = self.loss.loss_derivative(y,output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.learning_rate)

            epoch_loss /= len(inputs)
            if epoch % timestep == 0:
                print(f"Finished Epoch {epoch}, Loss {epoch_loss}")
                self.loss_history.append(epoch_loss)

    def predict(self,inputs : np.ndarray):
        data = inputs
        for layer in self.layers:
            data = layer.forward(data) 
        return data   
    def save(self, model_name):
        weights_file = f"{model_name}_weights.txt"
        biases_file = f"{model_name}_biases.txt"
        
        all_weights = []
        all_biases = []
        
        for layer in self.layers:
            all_weights.append(layer.weights.flatten())
            all_biases.append(layer.biases.flatten())
        
        np.savetxt(weights_file, np.concatenate(all_weights))
        np.savetxt(biases_file, np.concatenate(all_biases))
    
    def load(self, model_name):
        weights_file = f"{model_name}_weights.txt"
        biases_file = f"{model_name}_biases.txt"
        
        all_weights = np.loadtxt(weights_file)
        all_biases = np.loadtxt(biases_file)
        
        weight_offset = 0
        bias_offset = 0
        
        for layer in self.layers:
            weights_shape = layer.weights.shape
            biases_shape = layer.biases.shape
            
            weight_size = np.prod(weights_shape)
            bias_size = np.prod(biases_shape)
            
            layer.weights = all_weights[weight_offset:weight_offset + weight_size].reshape(weights_shape)
            layer.biases = all_biases[bias_offset:bias_offset + bias_size].reshape(biases_shape)
            
            weight_offset += weight_size
            bias_offset += bias_size
        
        
