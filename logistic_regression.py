#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def logistic_function(x, a=0, b=0):
    return 1 / (1 + np.exp(-(a * x + b)))


# binary cross-entropy loss
def bcel(y, y_hat):
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)  # To avoid log(0)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def read_data(path):
    dataframe = pd.read_csv(path, header=0)
    return dataframe


def plot(dataframe, a, b):
    #sns.scatterplot(data=dataframe, x='Pclass', y="Survived")
    #plt.show()

    sns.scatterplot(data=dataframe, x='Age', y='Survived')
    x_vals = np.linspace(dataframe['Age'].min(), dataframe['Age'].max(), 100)
    y_vals = logistic_function(x_vals, a, b)
    sns.lineplot(x=x_vals, y=y_vals, color='red', label=f'Logistic (a={a}, b={b})')

    plt.xlabel("Age")
    plt.ylabel("Survived Probability")
    plt.legend()
    plt.show()
    


def loss(dataframe, a, b):
    y_hat = logistic_function(dataframe['Age'], a, b)
    y_true = dataframe['Survived']
    print(bcel(y_true,y_hat))
    return bcel(y_true, y_hat)

df = read_data("titanic.csv") 

for a, b in [(0.2, -10), (0.2, -5), (0.2, -6), (0.2,-7)]:
    print(f"Loss with a={a}, b={b}: {loss(df, a, b)}")
    plot(df, a, b)