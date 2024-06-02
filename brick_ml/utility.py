import numpy as np
import brick_ml.losses
import brick_ml.layers
import brick_ml.activations
import importlib
def test_train_split(X: np.ndarray | list, y: np.ndarray | list, split_size: float):
    """
    Split the data into training and test sets.

    Args:
    - X (np.ndarray or list): Input data.
    - y (np.ndarray or list): True labels for the input data.
    - split_size (float): The proportion of the data to be used for training.

    Returns:
    - X_train (np.ndarray): Training data.
    - X_test (np.ndarray): Test data.
    - y_train (np.ndarray): True labels for the training data.
    - y_test (np.ndarray): True labels for the test data.

    Raises:
    - ValueError: If X and y are not the same size.
    """
    # Check if X and y have the same size
    if len(X) != len(y):
        raise ValueError("X and y must be the same size")
    
    # Get the indices of the data and shuffle them
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # Shuffle the data based on the indices
    shuffled_X = [X[i] for i in indices]
    shuffled_y = [y[i] for i in indices]
    
    # Calculate the split index based on the split size
    split_index = int(split_size * len(indices))
    
    # Split the data into training and test sets
    X_train = np.array(shuffled_X[:split_index])
    X_test =  np.array(shuffled_X[split_index:])
    y_train =  np.array(shuffled_y[:split_index])
    y_test =  np.array(shuffled_y[split_index:])
    
    return X_train, X_test, y_train, y_test

def vectorize(size, idx):
    """
    Converts an index to a binary vector representation.

    Args:
    - size (int): The size of the vector.
    - idx (int): The index to be converted.

    Returns:
    - arr (np.ndarray): The binary vector representation of the index.
    """
    # Create a zero array of shape (size,)
    arr = np.zeros(size, dtype=int)
    
    # Set the idx-th element of arr to 1.0
    arr[idx] = 1.0
    return arr

def get_loss(loss_name: str):
    """
    Returns the loss function based on the loss name.

    Args:
    - loss_name (str): The name of the loss function.

    Returns:
    - loss (brick_ml.losses.Loss): The loss function.
    """
    return getattr(importlib.import_module(f"brick_ml.losses.{loss_name}"), loss_name)

def get_layer(layer_name: str):
    """
    Returns the layer class based on the layer name.

    Args:
    - layer_name (str): The name of the layer.

    Returns:
    - layer (brick_ml.layers.Layer): The layer class.
    """
    return getattr(importlib.import_module(f"brick_ml.layers.{layer_name}"), layer_name)

def get_activation(activation_name: str):
    """
    Returns the activation function based on the activation name.

    Args:
    - activation_name (str): The name of the activation function.

    Returns:
    - activation (brick_ml.activations.Activation): The activation function.
    """
    return getattr(importlib.import_module(f"brick_ml.activations.{activation_name}"), activation_name)