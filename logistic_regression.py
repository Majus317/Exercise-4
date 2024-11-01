#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def logistic_function(x, a, b):
     y = 1/(1 + np.exp(-(a*x + b)))
     return y


# binary cross-entropy loss
def bcel(y, y_hat):
    bcel = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return np.mean(bcel)


def read_data(path):
    dataframe = pd.read_csv(path, header = 0)
    return dataframe


def plot(dataframe, a, b):
    scatterplot1 = sns.scatterplot(x='Age', y='Survived', data=dataframe)
    y_vals = logistic_function(dataframe['Age'], a=a, b=-b)
    sns.lineplot(x=dataframe['Age'], y=y_vals, ax=scatterplot1, color='red')
    plt.show()

def plot2(dataframe, a, b):
    scatterplot2 = sns.scatterplot(x= 'Pclass', y= 'Survived', data= dataframe)
    y_vals = logistic_function(dataframe['Pclass'], a=a, b=b)
    sns.lineplot(x=dataframe['Pclass'], y=y_vals, ax=scatterplot2, color='red')
    plt.show()


def loss(dataframe, a, b):
    y = dataframe['Survived']
    y_hat = logistic_function(dataframe['Age'], a=a, b=b)
    loss = bcel(y, y_hat)
    return loss

def loss2(dataframe, a, b):
    y = dataframe['Survived']
    y_hat = logistic_function(dataframe['Pclass'], a=a, b=b)
    loss = bcel(y, y_hat)
    return loss

dataframe = read_data('titanic.csv')

"""# Here are my minimum Losses"""
#plot(dataframe, a=0.01, b=-0.4)
#plot2(dataframe, a=0.5, b=-1.6)

for a, b in [(0.01, -0.4), (0.05, -0.4), (0.5, -0.4), (5,-0.4)]:
    plot(dataframe, a, b)

for a, b in [(0.5, -1.6), (0.5, -3), (0.5, -6)]:
    plot2(dataframe, a, b)

