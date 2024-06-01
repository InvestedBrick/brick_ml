import numpy as np
import sklearn.datasets # for dataset
import brick_ml.utility as util
import brick_ml.models.Sequential as Sequential
import brick_ml.layers.Dense as Dense
import brick_ml.losses.mse as mse
import brick_ml.activations.Sigmoid as Sigmoid
import brick_ml.activations.ReLU as ReLU
import brick_ml.activations.Tanh as Tanh
import matplotlib.pyplot as plt

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
    model.train(n_epochs=10000,timestep=100,inputs=X,expected_output=y)

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
    data_set = sklearn.datasets.load_iris()
    X = data_set["data"]
    y = [util.vectorize(size=3,idx=i) for i in data_set["target"]]
    names = data_set["target_names"]
    X_train,X_test,y_train,y_test = util.test_train_split(X,y,split_size=0.7)
    model = Sequential.Sequential(0.07,mse.MSE())
    model.add_layer(Dense.Dense(n_inputs=4,n_neurons=5,activation=Sigmoid.Sigmoid()))
    model.add_layer(Dense.Dense(n_inputs=model.last_layer_size,n_neurons=3,activation=Tanh.Tanh()))
    model.add_layer(Dense.Dense(n_inputs=model.last_layer_size,n_neurons=3,activation=Sigmoid.Sigmoid()))
    model.train(n_epochs=5000,timestep=100,inputs=X_train,expected_output=y_train)
    accuracy = model.evaluate(X_test,y_test)
    print(f"Accuracy: {accuracy*100:.2f} %")
    savemodel = input("Save Model(y/n)?")
    if savemodel == "y":
        model.save("iris")     
    
    
    xs = [x for x in range(len(model.loss_history))]
    plt.plot(xs,model.loss_history)
    plt.show()
    
def main():
    #parity()
    iris()

if __name__ == "__main__":
    main()
