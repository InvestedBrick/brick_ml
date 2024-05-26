import numpy as np
import brick_ml.models.Sequential as Sequential
import brick_ml.layers.Dense as Dense
import brick_ml.losses.mse as mse
import brick_ml.activations.Sigmoid as Sigmoid
import brick_ml.activations.ReLU as ReLU
import matplotlib.pyplot as plt

model = Sequential.Sequential(0.02,mse.MSE())
model.add_layer(Dense.Dense(n_inputs=3,n_neurons=3,activation=ReLU.ReLU()))
model.add_layer(Dense.Dense(n_inputs=3,n_neurons=1,activation=Sigmoid.Sigmoid()))

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

model.train(n_epochs=20000,timestep=500,inputs=X,expected_output=y)

print("Testing Model:")
print(f"[0,0]: {model.predict([0,0,1])}")#1
print(f"[1,0]: {model.predict([1,1,0])}")#0
print(f"[0,1]: {model.predict([1,0,1])}")#0
print(f"[1,1]: {model.predict([0,0,0])}")#0
model.save("parity")
xs = [x for x in range(len(model.loss_history))]

plt.plot(xs,model.loss_history)
plt.show()