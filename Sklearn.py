import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pickle

data = datasets.load_breast_cancer()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

"""best = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    model = KNeighborsClassifier(n_neighbors=21)
    model.fit(x_train, y_train)

    acc = model.score(x_test, y_test)

    print(acc)

    if acc > best:
        best = acc
        with open("firemodel.pickle", "wb") as f:
            pickle.dump(model, f)

print("The best one was ", best)"""

pickle_in = open("firemodel.pickle", "rb")
model = pickle.load(pickle_in)

predictions = model.predict(x_test)

for x in range(len(predictions)):
    print("Predict:", predictions[x], "  Actual:", y_test[x], "  Dataset", x_test[x])


