#!/usr/bin/env python3


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def logistic_function(x, a=0, b=0):
   y_hat= 1 / (1 + np.exp(-(a * x + b)))
   return y_hat


# binary cross-entropy loss
def bcel(y, y_hat):
   bcel= -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
   return bcel


def read_data(path):
    dataframe=pd.read_csv(path, header=0)
    
    return dataframe.dropna()



def plot(dataframe, x_axis):
    sns.scatterplot(x=x_axis, y="Survived", data=dataframe)
    x_values = np.linspace(dataframe[x_axis].min(), dataframe[x_axis].max(), 100)
    y_values = logistic_function(x_values, a=0.05, b=-3)
    sns.lineplot(x=x_values, y=y_values, color="red", label="Logistic Function")


def loss(dataframe, x_axis):
    x = dataframe[x_axis]
    y = dataframe["Survived"]
    y_hat = logistic_function(x, a=-7, b=4) 
    loss=bcel(y, y_hat)
    return loss
    


plot(read_data('titanic.csv'), "Age")
loss_value = loss(read_data('titanic.csv'), "Age")

plot(read_data('titanic.csv'), "Pclass")
loss_value = loss(read_data('titanic.csv'), "Pclass")