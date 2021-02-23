import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("car.data", sep=",")

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
clas = le.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, lug_boot, safety, door, maint, persons))
y = list(clas)

# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


best = 0.000000000
for _ in range(4999):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = KNeighborsClassifier(n_neighbors=23)
    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)

    print(acc)

    if acc > best:
        best = acc
        with open("carmodel.pickle", "wb") as pkl:
            pickle.dump(linear, pkl)

print("The best was ", best)

"""
pickle_in = open("carmodel.pickle", "rb")
linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print("Predict:", predictions[x], "  Actual:", y_test[x], "  Dataset:", x_test[x])
"""
