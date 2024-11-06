#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def logistic_function(x, a=0, b=0):
    # np.exp is a monadic function of "e to the power of n"
    return 1 / (1 + np.exp(-(a*x+b)))


# binary cross-entropy loss
def bcel(y, y_hat):
    return y*np.log(y_hat) + ((1 - y)*np.log(1-y_hat))


def read_data(path):
    return pd.read_csv(path, header=0)


def plot(dataframe):
    # 1.
    plt.subplot(1, 2, 1)
    sns.scatterplot(dataframe, x="Age",y="Survived")
    plt.title("Age-Survived")

    # 2.#
    plt.subplot(1, 2, 2)
    sns.scatterplot(dataframe, x="Pclass", y="Survived")
    plt.title("Class-Survived")

    plt.show()


def loss(dataframe):
    pass

df = read_data("titanic.csv")
plot(df)
