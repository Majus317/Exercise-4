#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def logistic_function(x, a=0, b=0):
    database = pd.read_csv('titanic.csv', header = 0) 
    #print(database)
    sns.scatterplot(data = database, x = x, y = 'Survived')
    plt.show()
    



# binary cross-entropy loss
def bcel(y, y_hat):
    pass


def read_data(path):
    pass


def plot(dataframe):
    pass


def loss(dataframe):
    pass

logistic_function(x = "Age")
logistic_function(x = "Pclass")