#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def logistic_function(x, a=0, b=0):
    y_hat = 1/(1 + np.exp(a*x+b))
    return y_hat
    
# binary cross-entropy loss
def bcel(y, y_hat):
    return np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def read_data(path):
     dataframe = pd.read_csv(path, header = 0) 
     #print(database)
     return dataframe


def plot(dataframe,  a_value, b_value):
    plt.subplot(1,2,1)
    sns.scatterplot(data = dataframe, x = "Age", y = 'Survived')

    x = dataframe["Age"]
    sns.lineplot(x= x, y= logistic_function(x, a_value, b_value), color = 'green')

    plt.subplot(1,2,2)
    sns.scatterplot(data = dataframe, x = "Pclass", y = 'Survived')
    plt.show()


def loss(dataframe , a_value, b_value):
    y = dataframe["Survived"]
    y_hat = logistic_function(dataframe["Age"], a_value , b_value)
    return bcel(y, y_hat)

print(loss(read_data('titanic.csv'), a_value = 1, b_value = 1))

plot(read_data('titanic.csv'), a_value = 1, b_value = 1)
