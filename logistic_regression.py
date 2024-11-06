#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def logistic_function(x, a, b):
    return 1 / (1 + np.exp(-(a * x + b)))


# Binary cross-entropy loss
def bcel(y, y_hat):
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)  
    return np.mean(-(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


def read_data(path):
    dataframe = pd.read_csv(path, header=0)
    return dataframe


def plot(dataframe, a, b):
    plt.figure()
    sns.scatterplot(x='Age', y='Survived', data=dataframe)
    y_vals = logistic_function(dataframe['Age'], a=a, b=b)
    sns.lineplot(x=dataframe['Age'], y=y_vals, ax=plt.gca(), color='red')
    plt.show()


def plot2(dataframe, a, b):
    plt.figure()
    sns.scatterplot(x='Pclass', y='Survived', data=dataframe)
    y_vals = logistic_function(dataframe['Pclass'], a=a, b=b)
    sns.lineplot(x=dataframe['Pclass'], y=y_vals, ax=plt.gca(), color='red')
    plt.show()


def loss(dataframe, a, b):
    y = dataframe['Survived']
    y_hat = logistic_function(dataframe['Age'], a=a, b=b)
    return bcel(y, y_hat)


def loss2(dataframe, a, b):
    y = dataframe['Survived']
    y_hat = logistic_function(dataframe['Pclass'], a=a, b=b)
    return bcel(y, y_hat)


dataframe = read_data('titanic.csv')
print(dataframe[['Age', 'Pclass', 'Survived']].isna().sum())


# Trying different values of a and b for Age vs. Survived plot
for a, b in [(0.01, -0.4), (0.05, -0.4), (0.5, -0.4), (0.01, -0.84)]:
    plot(dataframe, a, b)

# Trying different values of a and b for Pclass vs. Survived plot
for a, b in [(0.5, -1.6), (0.5, -3), (0.5, -6), (0.01, -0.47)]:
    plot2(dataframe, a, b)

#Optimal values for age: a = 0.01, b = -0.84, Minimum Loss = 0.987

#Optimal values for Pclass: a = 0.01, b = -0.47, Minimum Loss = 0.667 

