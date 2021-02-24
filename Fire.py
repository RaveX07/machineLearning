import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pickle
import sklearn
import matplotlib
from matplotlib import style

data = pd.read_csv("forestfires.csv", sep=",")

predict = "area"

le = preprocessing.LabelEncoder()
month = le.fit_transform(list(data["month"]))
DMC = le.fit_transform(list(data["DMC"]))
DC = le.fit_transform(list(data["DC"]))
ISI = le.fit_transform(list(data["ISI"]))
temp = le.fit_transform(list(data["temp"]))
wind = le.fit_transform(list(data["wind"]))
rain = le.fit_transform(list(data["rain"]))
area = le.fit_transform(list(data["area"]))
RH = le.fit_transform(list(data["RH"]))
X = le.fit_transform(list(data["X"]))
Y = le.fit_transform(list(data["Y"]))
FFMC = le.fit_transform(data["FFMC"])


x = list(zip(month, temp, rain, X, Y, DMC, DC, ISI, FFMC))
y = list(area)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


style.use("ggplot")
p = "FFMC"
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel(predict)
pyplot.show()


best = 0
for _ in range(4999):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    model = KNeighborsClassifier(n_neighbors=21)
    model.fit(x_train, y_train)

    acc = model.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("firemodel.pickle", "wb") as f:
            pickle.dump(model, f)

print("The best was ", best)
"""

pickle_in = open("firemodel.pickle", "rb")
model = pickle.load(pickle_in)

predictions = model.predict(x_test)

for x in range(len(predictions)):
    print("Predict:", predictions[x], "  Actual:", y_test[x], "  Dataset:", x_test[x])
"""