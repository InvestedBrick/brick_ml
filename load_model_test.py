import brick_ml.models.Sequential as Sequential
import sklearn.datasets
import brick_ml.utility as util

data_set = sklearn.datasets.load_iris()
X = data_set["data"]
y = [util.vectorize(size=3,idx=i) for i in data_set["target"]]
names = data_set["target_names"]
X_train,X_test,y_train,y_test = util.test_train_split(X,y,split_size=0.7)

model = Sequential.Sequential(None,None)
model.load("iris")

accuracy = model.evaluate(X_test,y_test)
print(f"Accuracy: {accuracy*100:.2f} %")
