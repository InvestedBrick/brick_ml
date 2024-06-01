import numpy as np

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
    # Create a zero array of shape (1, size)
    arr = np.zeros((1, size), dtype=int)
    
    # Set the idx-th element of arr to 1.0
    arr[0][idx] = 1.0
    
    # Return the array
    return arr
