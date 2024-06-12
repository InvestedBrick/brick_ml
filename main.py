import numpy as np
import matplotlib.pyplot as plt

import sklearn.datasets as sklearn_data# for dataset
import keras.datasets  as keras_data# for dataset

import brick_ml.utility as util

import brick_ml.models.Sequential as Sequential

import brick_ml.layers.Dense as Dense
import brick_ml.layers.Convolutional as Convolutional
import brick_ml.layers.Reshape as Reshape
import brick_ml.layers.Pooling as Pooling
import brick_ml.layers.Softmax as Softmax

import brick_ml.losses.mse as mse
import brick_ml.losses.binary_cross_entropy as binary_cross_entropy

import brick_ml.activations.Sigmoid as Sigmoid
import brick_ml.activations.Tanh as Tanh
import brick_ml.activations.ReLU as ReLU

def parity_check(data):
    """
    Checks if the sum of the elements in the data array is odd.
    """
    return np.sum(data) % 2 != 0

def parity():
    model = Sequential.Sequential(0.2,mse.MSE())
    model.add_layer(Dense.Dense(n_inputs=3,n_neurons=5,activation=Sigmoid.Sigmoid()))
    model.add_layer(Dense.Dense(n_inputs=model.last_layer_size,n_neurons=1,activation=Sigmoid.Sigmoid()))

    X = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    #0, 1, 1, 0, 1, 0, 0, 1
    y = np.array([[0],[1],[1],[0],[1],[0],[0],[1]])
    model.train(n_epochs=10000,timestep=100,inputs=X,expected_output=y,batch_size=None)

    print("Testing Model:")
    off = 0
    for x in X:
        pred = model.predict(x)
        validation = parity_check(x)
        print(f"[{x}]: {pred} | {validation}, off by: {abs(pred - validation)}")
        off += abs(pred - validation)
    print(f"Total off: {off}")    

    choice = input("Save Model(y/n)?")

    if choice == "y":
        model.save("parity")
    
    xs = [x for x in range(len(model.loss_history))]
    plt.plot(xs,model.loss_history)
    plt.show()
def iris():
    data_set = sklearn_data.load_iris()
    X = data_set["data"]
    y = [util.vectorize(size=3,idx=i) for i in data_set["target"]]
    X_train,X_test,y_train,y_test = util.test_train_split(X,y,split_size=0.7)
    model = Sequential.Sequential(0.07,mse.mse())
    model.add_layer(Dense.Dense(n_inputs=4,n_neurons=5,activation=Sigmoid.Sigmoid()))
    model.add_layer(Dense.Dense(n_inputs=model.last_layer.n_neurons,n_neurons=3,activation=Tanh.Tanh()))
    model.add_layer(Dense.Dense(n_inputs=model.last_layer.n_neurons,n_neurons=3,activation=Sigmoid.Sigmoid()))
    model.train(n_epochs=5000,timestep=100,inputs=X_train,expected_output=y_train,batch_size=12,shuffle=True)
    accuracy = model.evaluate(X_test,y_test)
    print(f"Accuracy: {accuracy*100:.2f} %")
    savemodel = input("Save Model(y/n)?")
    if savemodel == "y":
        model.save("iris")     
    
    
    xs = [x for x in range(len(model.loss_history))]
    plt.plot(xs,model.loss_history)
    plt.show()

def mnist():
    (X_train, y_train), (X_test, y_test) = keras_data.mnist.load_data()
    
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255 

    X_train = X_train.reshape(len(X_train), 1, 28, 28)
    X_test = X_test.reshape(len(X_test),1, 1, 28, 28) # reshape for a btach size of 1 for evaluation


    y_train = np.array([util.vectorize(size=10,idx=i) for i in y_train])
    y_test = np.array([util.vectorize(size=10,idx=i) for i in y_test])
    
    y_train = y_train.reshape(len(y_train),10)
    y_test = y_test.reshape(len(y_test),1,10)


    model = Sequential.Sequential(0.1,binary_cross_entropy.binary_cross_entropy())
    model.add_layer(Convolutional.Convolutional(input_shape=(1,28,28),kernel_size=3,n_kernels=8,activation=Tanh.Tanh()))
    model.add_layer(Pooling.Pooling(pool_size=(2,2),stride=2,pad=0,pool_function="max"))
    vector_size = 8 * 13 * 13 
    model.add_layer(Reshape.Reshape(input_shape=(16,8,13,13),output_shape=(vector_size,),add_batch_size=False)) # input_shape = batch_size, n_kernels, pool_output_width, pool_output_height
    model.add_layer(Dense.Dense(n_inputs=vector_size,n_neurons=32,activation=Tanh.Tanh()))
    model.add_layer(Dense.Dense(n_inputs=model.last_layer.n_neurons,n_neurons=10,activation=None))
    model.add_layer(Softmax.Softmax())

    model.train(n_epochs=50,timestep=1,inputs=X_train,expected_output=y_train,batch_size=16,shuffle=True)

    accuracy = model.evaluate(X_test,y_test)
    print(f"Accuracy: {accuracy*100:.2f} %")
    xs = [x for x in range(len(model.loss_history))]
    plt.plot(xs,model.loss_history)
    plt.show()

def main():
    #parity()
    #iris()
    mnist()

if __name__ == "__main__":
    main()