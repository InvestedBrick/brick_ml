import numpy as np
import json
import brick_ml.utility as util
import time
class Sequential:
    def __init__(self,learning_rate : float, loss,learning_rate_decay = 1) -> None:
        self.layers = []
        self.loss_history = []
        self.learning_rate = learning_rate
        self.loss = loss
        self.learning_rate_decay = learning_rate_decay
    def add_layer(self, layer):
        """
        Adds a layer to the model.

        Args:
        - layer: An instance of a layer class.
        """
        self.layers.append(layer)  # Append the layer to the list of layers
        self.last_layer = self.layers[-1]  # Update the last layer

    def train(self, n_epochs: int, timestep: int, inputs: np.ndarray, expected_output: np.ndarray, batch_size: int | None, shuffle: bool = True):
        """
        Trains the model for a given number of epochs.

        Args:
        - n_epochs (int): Number of epochs to train the model for.
        - timestep (int): Number of epochs to wait before printing the loss.
        - inputs (np.ndarray): Input data.
        - expected_output (np.ndarray): True labels for the input data.
        - batch_size (int): Size of the mini-batch.
        - shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        """
        # Set default batch size if None or negative
        if batch_size is None or batch_size < 0:
            batch_size = len(inputs)

        # Train the model for the specified number of epochs
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            # Shuffle the data if specified
            if shuffle:
                # Shuffle the data indices
                indices = np.arange(len(inputs))
                np.random.shuffle(indices)
                # Shuffle the input and expected output arrays
                inputs = np.array([inputs[i] for i in indices])
                expected_output = np.array([expected_output[i] for i in indices])

            epoch_loss = 0.0  # Accumulate the loss for the epoch

            # Iterate over each mini-batch
            for i in range(0, len(inputs), batch_size):
                # Get the mini-batch
                X_batch = inputs[i:i+batch_size]
                y_batch = expected_output[i:i+batch_size]

                # Predict the output for the mini-batch
                output = self.predict(X_batch, training=True)
                # Calculate the loss
                loss = self.loss.loss(y_batch, output)
                epoch_loss += loss
                
                # Calculate the gradient of the loss with respect to the weights
                gradient = self.loss.loss_derivative(y_batch, output)
                # Backpropagate the gradient through the layers
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.learning_rate)

            # Calculate the average loss for the epoch
            epoch_loss /= len(inputs) // batch_size
            self.learning_rate *= self.learning_rate_decay
            # Print the loss if the epoch is a multiple of timestep
            if epoch % timestep == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"Finished Epoch {epoch}, Loss {epoch_loss}, epoch took {epoch_time} seconds to complete (~{((n_epochs - (epoch + 1)) * epoch_time) / 60:.2f} minutes remaining)")
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
            predicted_label = np.argmax(self.predict(x))
            # If the predicted label equals the true label, increments the counter.
            if predicted_label == np.argmax(y):
                counter += 1
        # Returns the accuracy of the model on the test data.
        return counter / len(X_test) 

    
    #update to work for Dense, Convolutional,Dropout,Pooling,Rehsape and Softmax
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
            if layer.__class__.__name__ == "Dense":
                all_weights.append(layer.weights.flatten())
                all_biases.append(layer.biases.flatten())
            elif layer.__class__.__name__ == "Convolutional":
                all_weights.append(layer.kernels.flatten())
                all_biases.append(layer.biases.flatten())
                  
        
        # Save the concatenated weights and biases to the respective text files.
        np.savetxt(weights_file, np.concatenate(all_weights))
        np.savetxt(biases_file, np.concatenate(all_biases))
        
        # Store layer data, batch size, and learning rate in a dictionary.
        data = {}
        data["learning_rate"] = self.learning_rate
        data["loss"] = self.loss.__class__.__name__
        data["lr_decay"] = self.learning_rate_decay
        layers = {}
        
        # Iterate over each layer and store its data in the dictionary.
        for i, layer in enumerate(self.layers):
            layer_data = {}
            if layer.__class__.__name__ == "Dense":
                layer_data["n_inputs"] = layer.n_inputs
                layer_data["n_neurons"] = layer.n_neurons
                layer_data["activation"] = layer.activation.__class__.__name__
            elif layer.__class__.__name__ == "Convolutional":
                    layer_data["input_shape"] = layer.input_shape
                    layer_data["kernel_size"] = layer.kernel_size
                    layer_data["n_kernels"] = layer.n_kernels
                    layer_data["activation"] = layer.activation.__class__.__name__
            elif layer.__class__.__name__ == "Dropout":
                layer_data["dropout_rate"] = layer.dropout_rate
            elif layer.__class__.__name__ == "Pooling":
                layer_data["pool_size"] = layer.pool_size
                layer_data["stride"] = layer.stride
                layer_data["pad"] = layer.pad
                layer_data["pool_function"] = layer.pool_function
            elif layer.__class__.__name__ == "Reshape":
                layer_data["input_shape"] = layer.input_shape
                layer_data["output_shape"] = layer.output_shape     
                    
            layer_data["type"] = layer.__class__.__name__
            layers[f"layer_{i}"] = layer_data
        
        data["layers"] = layers    
        
        # Save the metadata (layer data, batch size, learning rate) to a JSON file.
        with open(f"{model_name}_metadata.json", "w",encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    
    def load(self, model_name):
        """
        Loads the model weights and biases from separate text files.

        Args:
        - model_name (str): The name of the model.
        """
        # Define the file names for the weights and biases.
        weights_file = f"{model_name}_weights.txt"
        biases_file = f"{model_name}_biases.txt"
        metadata_file = f"{model_name}_metadata.json"
        # Load the weights and biases from the respective text files.
        all_weights = np.loadtxt(weights_file)
        all_biases = np.loadtxt(biases_file)
        with open(metadata_file, "r",encoding="utf-8") as f:
            data = json.load(f)
        self.learning_rate = data["learning_rate"]
        self.loss = util.get_loss(data["loss"])
        self.learning_rate_decay = data["lr_decay"]
        # Initialize offsets for the weights and biases.
        weight_offset = 0
        bias_offset = 0
        layers = data["layers"] 
        for  layer_data in layers.values():
            if layer_data["type"] == "Dense":
                layer = util.get_layer(layer_data["type"])(
                n_inputs=layer_data["n_inputs"], 
                n_neurons=layer_data["n_neurons"], 
                activation=util.get_activation(layer_data["activation"])()
                )
            elif layer_data["type"] == "Convolutional":
                layer = util.get_layer(layer_data["type"])(
                input_shape=tuple(layer_data["input_shape"]), 
                kernel_size=layer_data["kernel_size"], 
                n_kernels=layer_data["n_kernels"], 
                activation=util.get_activation(layer_data["activation"])()
                )
            elif layer_data["type"] == "Dropout":
                layer = util.get_layer(layer_data["type"])(
                dropout_rate=layer_data["dropout_rate"]
                )    
            elif layer_data["type"] == "Pooling":
                layer = util.get_layer(layer_data["type"])(
                pool_size=tuple(layer_data["pool_size"]), 
                stride=layer_data["stride"], 
                pad=layer_data["pad"],
                pool_function=layer_data["pool_function"]
                )    
            elif layer_data["type"] == "Reshape":
                layer = util.get_layer(layer_data["type"])(
                input_shape=tuple(layer_data["input_shape"]), 
                output_shape=tuple(layer_data["output_shape"])
                )
            elif layer_data["type"] == "Softmax":
                layer = util.get_layer(layer_data["type"])()
            self.layers.append(layer)
        # Iterate over each layer.
        for layer in self.layers:
            # Get the shapes of the weights and biases for the layer.
            if layer.__class__.__name__ == "Dense":
                weights_shape = layer.weights.shape
                biases_shape = layer.biases.shape
            elif layer.__class__.__name__ == "Convolutional":
                weights_shape = layer.kernels.shape
                biases_shape = layer.biases.shape
            else:
                continue
            
            # Calculate the size of the weights and biases for the layer.
            weight_size = np.prod(weights_shape)
            bias_size = np.prod(biases_shape)
            # Reshape the weights and biases from the flattened arrays and assign them to the layer.
            if layer.__class__.__name__ == "Dense":
                layer.weights = all_weights[weight_offset:weight_offset + weight_size].reshape(weights_shape)
                layer.biases = all_biases[bias_offset:bias_offset + bias_size].reshape(biases_shape)
            elif layer.__class__.__name__ == "Convolutional":
                layer.kernels = all_weights[weight_offset:weight_offset + weight_size].reshape(weights_shape)
                layer.biases = all_biases[bias_offset:bias_offset + bias_size].reshape(biases_shape)
            
            # Update the offsets for the weights and biases.
            weight_offset += weight_size
            bias_offset += bias_size