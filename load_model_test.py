import brick_ml.models.Sequential as Sequential
import brick_ml.utility as util
import numpy as np
import keras.datasets  as keras_data

(X_train, y_train), (X_test, y_test) = keras_data.mnist.load_data()

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255 
X_train = X_train.reshape(len(X_train), 28 * 28,)
X_test = X_test.reshape(len(X_test),1, 28 * 28,) # reshape for a btach size of 1 for evaluation
y_train = np.array([util.vectorize(size=10,idx=i) for i in y_train])
y_test = np.array([util.vectorize(size=10,idx=i) for i in y_test])

y_train = y_train.reshape(len(y_train),10)
y_test = y_test.reshape(len(y_test),1,10)

model = Sequential.Sequential(None,None)
model.load("mnist_dense")

accuracy = model.evaluate(X_test,y_test)
print(f"Accuracy: {accuracy*100:.2f} %")
