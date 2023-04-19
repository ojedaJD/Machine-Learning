import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
'''print(data.head())'''

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
'''print(data.head())'''

#label for training data
predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

#After 100 iterations of training. we will not be training any more models
'''best = 0
for _ in range(100):
#cannot train model off of testing data because it has been exposed to the data already and thus we are using only 10% of the data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

#using x_train and y_train data to create a best fit line
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

#testing the accuracy of the model  
    print(acc)

#only going to save a new model if the current score is betetr than any pervious score that we have seen, and it will go through 100 iterations of training
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)'''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

#what are the m and b values? From this we see the second coefficient has the heaviest weight
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#p can be changed to get different points for the scatterplot and see varying correlations between final grade and x-variable
p = 'absences'
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
