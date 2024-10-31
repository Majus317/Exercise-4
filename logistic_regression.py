#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def logistic_function(x, a=0, b=0):
    pass


# binary cross-entropy loss
def bcel(y, y_hat):
    pass


def read_data(path):
    dataframe = pd.read_csv(path, header=0)
    return dataframe


def plot(dataframe):
    sns.scatterplot(data=dataframe, x='Age', y='Survived')
    plt.show()
    sns.scatterplot(data=dataframe, x='Pclass', y="Survived")
    plt.show()


def loss(dataframe):
    pass

df = read_data("titanic.csv") 
plot(df)