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
    y_clip = np.clip(y_hat, 1e-15, 1 - 1e-15)
    return np.mean(-(y * np.log(y_clip) + (1 - y) * np.log(1 - y_clip)))


def read_data(path):
    return pd.read_csv(path, header=0)


def plot(dataframe, a, b):
    x_vals1 = np.linspace(dataframe["Age"].min(), dataframe["Age"].max(), 100)
    y_vals1 = logistic_function(x_vals1, a, b)
    sp1 = sns.scatterplot(data=dataframe, x="Age", y="Survived")
    lp1 = sns.lineplot(x=x_vals1, y=y_vals1, color="yellow")
    plt.title("Age & Survived")
    plt.show()

    x_vals2 = np.linspace(dataframe["Pclass"].min(), dataframe["Pclass"].max(), 100)
    y_vals2 = logistic_function(x_vals2, a, b)
    sp2 = sns.scatterplot(data=dataframe, x="Pclass", y="Survived")
    lp2 = sns.lineplot(x=x_vals2, y=y_vals2, color="yellow")
    plt.title("Passengerclass & Survived")
    plt.show()


def loss(dataframe, a, b):
    y_true = dataframe["Survived"]
    y_hat = logistic_function(dataframe["Age"], a, b)
    return bcel(y_true, y_hat)


def main():
    df = read_data("titanic.csv")

    # Testing to minimize the loss
    # for a, b in [(0.1, 1), (0.02, 2), (0.3, -0.5), (0.09,-1.5)]:
    # print(f"Loss for a={a}, b={b}: {loss(df, a, b)}")

    a = 0.09
    b = -1.5
    plot(df, a, b)
    print(f"The loss is: {loss(df,a,b)}")


# run main
main()
