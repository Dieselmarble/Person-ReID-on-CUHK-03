# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:51:11 2018

@author: zl6415
"""

%matplotlib inline

import metric_learn
import numpy as np
from sklearn.datasets import load_iris

# visualisation imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# loading our dataset

iris_data = load_iris()
# this is our data
X = iris_data['data']
# these are our constraints
Y = iris_data['target']

# function to plot the results
def plot(X, Y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    plt.figure(2, figsize=(8, 6))

    # clean the figure
    plt.clf()

    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.show()


# setting up LMNN
lmnn = metric_learn.LMNN(k=5, learn_rate=1e-6)

# fit the data!
lmnn.fit(X, Y)

# transform our input space
X_lmnn = lmnn.transform()
lmnn.metric()