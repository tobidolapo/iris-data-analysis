from math import *
import pandas as pd  # Import the pandas library
# import the train_test_split module from the Scikit library
from sklearn.model_selection import train_test_split
# Import the KNN classifier from Scikit library
from sklearn.neighbors import KNeighborsClassifier
# Import the accuracy module from from Scikit library
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Import root directory path from config for better relative path resolution
from config.definitions import ROOT_DIR
import os
import numpy as np

# Import the dataset
iris = pd.read_excel(os.path.join(ROOT_DIR, 'data\\raw',
                                  'iris.xlsx'))
print(iris.head())
# features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']

# Since we have categorical labels, KNN doesn't accept string labels. So we need to transform them into numbers
# It will return 0 for Setosa, 1 for Versicolor and 2 for Virginica
variety_num = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
iris['variety'] = iris['variety'].map(variety_num)

# Split the dataset into training and test dataset
# This means everyother column except the 'variety' column is a feature
features = iris.drop('variety', axis=1)
labels = iris['variety']  # This means 'variety' is a label or target

# x = features, y = labels. I am setting aside 60% for training set and 40% for test set.
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2)
# This is to check if the splitting was properly done
print(len(labels), len(x_train), len(x_test))

# Fit Classifier to the training set
# Instantiate the learning model at k =7
clsf = KNeighborsClassifier(n_neighbors=6).fit(x_train, y_train)

# The distance criterion chosen was euclidean distance


def euclidean_distance():
    return sqrt(sum(pow(a - b, 2) for a, b in zip(clsf.predict(x_test), y_test)))


print(euclidean_distance())  # Prints out the value of the euclidean distance

# Make predictions on the test data
y_pred = clsf.predict(x_test)
# Calculate the accuracy of the model
print("The accuracy of the model is:", str(
    round((accuracy_score(y_test, clsf.predict(x_test))), 3)) + '%\n')

# This prints the confusion matrix (Performance measurement for ML classifications)
print(confusion_matrix(y_test, y_pred))
# This prints the report of the classification
print(classification_report(y_test, y_pred), '\n')

# To visualize our accuracy score of 93%
print(y_pred, '\n')
y_expect = np.ravel(y_test)
print(y_expect)
print(len(y_expect))
