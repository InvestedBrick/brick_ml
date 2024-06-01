import numpy as np
class Sequential:
    def __init__(self,learning_rate : float, loss) -> None:
        self.layers = []
        self.loss_history = []
        self.learning_rate = learning_rate
        self.loss = loss
    def add_layer(self, layer):
        """
        Adds a layer to the model.

        Args:
        - layer: An instance of a layer class.
        """
        self.layers.append(layer)  # Append the layer to the list of layers
        self.last_layer_size = self.layers[-1].n_neurons  # Update the last layer size
    def train(self, n_epochs: int, timestep: int, inputs: np.ndarray, expected_output: np.ndarray):
        """
        Trains the model for a given number of epochs.

        Args:
        - n_epochs (int): Number of epochs to train the model for.
        - timestep (int): Number of epochs to wait before printing the loss.
        - inputs (np.ndarray): Input data.
        - expected_output (np.ndarray): True labels for the input data.
        """
        # Train the model for the specified number of epochs
        for epoch in range(n_epochs):
            epoch_loss = 0.0  # Accumulate the loss for the epoch

            # Iterate over each input and expected output pair
            for x,y in zip(inputs,expected_output):
                # Predict the output for the input
                output = self.predict(x.reshape(1,-1),training=True)

                # Calculate the loss
                loss = self.loss.loss(y,output)
                epoch_loss += loss

                # Calculate the gradient of the loss with respect to the weights
                gradient = self.loss.loss_derivative(y,output)

                # Backpropagate the gradient through the layers
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.learning_rate)

            # Calculate the average loss for the epoch
            epoch_loss /= len(inputs)

            # Print the loss if the epoch is a multiple of timestep
            if epoch % timestep == 0:
                print(f"Finished Epoch {epoch}, Loss {epoch_loss}")
                self.loss_history.append(epoch_loss)

    def predict(self, inputs: np.ndarray, training: bool = False):
        """
        Predicts the output of the model for the given inputs.

        Args:
        - inputs (np.ndarray): Input data.
        - training (bool, optional): Whether the model is in training mode. Defaults to False.

        Returns:
        - np.ndarray: Predicted output.
        """
        # Initialize the data with the input values.
        data = inputs

        # Iterate over each layer in the model.
        for layer in self.layers:
            # Pass the data through the layer and update the data.
            data = layer.forward(data, training) 

        # Return the final output of the model.
        return data  
    
    def evaluate(self,X_test : np.ndarray,y_test : np.ndarray):
        """
        Evaluates the model's accuracy on the test data.

        Args:
        - X_test (np.ndarray): Input test data.
        - y_test (np.ndarray): True labels for the test data.

        Returns:
        - float: The accuracy of the model on the test data.
        """
        # Counts the number of correct predictions.
        counter = 0
        # Iterates over each test data instance.
        for x,y in zip(X_test,y_test):
            # Predicts the label of the input data.
            predicted_label = np.argmax(self.predict(x.reshape(1,-1)))
            # If the predicted label equals the true label, increments the counter.
            if predicted_label == np.argmax(y):
                counter += 1
        # Returns the accuracy of the model on the test data.
        return counter / len(X_test) 

    
    def save(self, model_name):
        """
        Saves the model weights and biases to separate text files.

        Args:
        - model_name (str): The name of the model.
        """
        # Define the file names for the weights and biases.
        weights_file = f"{model_name}_weights.txt"
        biases_file = f"{model_name}_biases.txt"
        
        # Initialize lists to store the flattened weights and biases.
        all_weights = []
        all_biases = []
        
        # Iterate over each layer and append the flattened weights and biases.
        for layer in self.layers:
            all_weights.append(layer.weights.flatten())
            all_biases.append(layer.biases.flatten())
        
        # Save the concatenated weights and biases to the respective text files.
        np.savetxt(weights_file, np.concatenate(all_weights))
        np.savetxt(biases_file, np.concatenate(all_biases))
    
    def load(self, model_name):
        """
        Loads the model weights and biases from separate text files.

        Args:
        - model_name (str): The name of the model.
        """
        # Define the file names for the weights and biases.
        weights_file = f"{model_name}_weights.txt"
        biases_file = f"{model_name}_biases.txt"
        
        # Load the weights and biases from the respective text files.
        all_weights = np.loadtxt(weights_file)
        all_biases = np.loadtxt(biases_file)
        
        # Initialize offsets for the weights and biases.
        weight_offset = 0
        bias_offset = 0
        
        # Iterate over each layer.
        for layer in self.layers:
            # Get the shapes of the weights and biases for the layer.
            weights_shape = layer.weights.shape
            biases_shape = layer.biases.shape
            
            # Calculate the size of the weights and biases for the layer.
            weight_size = np.prod(weights_shape)
            bias_size = np.prod(biases_shape)
            
            # Reshape the weights and biases from the flattened arrays and assign them to the layer.
            layer.weights = all_weights[weight_offset:weight_offset + weight_size].reshape(weights_shape)
            layer.biases = all_biases[bias_offset:bias_offset + bias_size].reshape(biases_shape)
            
            # Update the offsets for the weights and biases.
            weight_offset += weight_size
            bias_offset += bias_size
            bias_offset += bias_size
        
        
