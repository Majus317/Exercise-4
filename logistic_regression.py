#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def logistic_function(x, a=0, b=0):
    return 1 / (1 + np.exp(-(a * x + b)))

def bcel(y, y_hat):
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)  
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def read_data(path):
    return pd.read_csv(path)

def plot(dataframe):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=dataframe, x='Age', y='Survived', alpha=0.5)
    plt.title('Scatterplot of Age vs. Survival')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=dataframe, x='Pclass', y='Survived', alpha=0.5)
    plt.title('Scatterplot of Passenger Class vs. Survival')
    plt.show()

def plot_logistic(dataframe, x_col, a, b):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=dataframe, x=x_col, y='Survived', alpha=0.5)
    
    x_vals = np.linspace(dataframe[x_col].min(), dataframe[x_col].max(), 100)
    y_vals = logistic_function(x_vals, a, b)
    
    sns.lineplot(x=x_vals, y=y_vals, color='red')
    plt.title(f'Logistic Function Fit with a={a}, b={b}')
    plt.show()

def loss(dataframe, x_col, a, b):
    x_vals = dataframe[x_col]
    y_true = dataframe['Survived']
    y_pred = logistic_function(x_vals, a, b)
    
    return bcel(y_true, y_pred)

if __name__ == "__main__":
     path = 'path/to/titanic.csv'   
    df = read_data(path)
    
     plot(df)

     a_age, b_age = 0.1, -5  
    a_pclass, b_pclass = -1, 2  
    
    print("Age vs. Survival Logistic Plot and Loss:")
    plot_logistic(df, 'Age', a_age, b_age)
    loss_age = loss(df, 'Age', a_age, b_age)
    print(f"Binary Cross-Entropy Loss for Age vs. Survival: {loss_age:.4f}")
    
    print("\nPassenger Class vs. Survival Logistic Plot and Loss:")
    plot_logistic(df, 'Pclass', a_pclass, b_pclass)
    loss_pclass = loss(df, 'Pclass', a_pclass, b_pclass)
    print(f"Binary Cross-Entropy Loss for Passenger Class vs. Survival: {loss_pclass:.4f}")
