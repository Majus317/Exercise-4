#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using Titanic data.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 1. Define the logistic function
def logistic_function(x, a=0, b=0):
    """
    Logistic function: 1 / (1 + exp(-(a * x + b)))
    """
    return 1 / (1 + np.exp(-(a * x + b)))

# 2. Define the Binary Cross-Entropy Loss (BCE)
def bcel(y, y_hat):
    """
    Binary Cross-Entropy Loss: -1/N * sum(y * log(y_hat) + (1 - y) * log(1 - y_hat))
    """
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)  # Avoid log(0) issues
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# 3. Read the data
def read_data(path):
    """
    Reads the Titanic CSV file into a DataFrame
    """
    return pd.read_csv(path)

# 4. Plot the data
def plot(dataframe):
    """
    Create scatter plots of age vs survival and class vs survival.
    """
    plt.figure(figsize=(12, 5))

    # Plot 1: Age vs Survived
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Age', y='Survived', data=dataframe)
    plt.title("Age vs Survival")

    # Plot 2: Passenger Class vs Survived
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='Pclass', y='Survived', data=dataframe)
    plt.title("Passenger Class vs Survival")

    plt.show()

# 5. Calculate the loss
def loss(dataframe, a, b):
    """
    Calculate the binary cross-entropy loss for given values of a and b.
    """
    # Predict survival probability with logistic function
    y_hat_age = logistic_function(dataframe['Age'].fillna(0), a, b)
    y = dataframe['Survived']

    # Calculate the loss
    return bcel(y, y_hat_age)

# Main function to visualize and calculate loss
def main():
    # Read the data
    df = read_data('titanic.csv')

    # Plot the data
    plot(df)

    # Example values for a and b
    a, b = 0.05, -1  # Initial values

    # Calculate and print the loss
    current_loss = loss(df, a, b)
    print(f"Binary Cross-Entropy Loss with a={a}, b={b}: {current_loss}")

    # Plot logistic function on age vs survival scatter plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='Age', y='Survived', data=df)
    ages = np.linspace(0, 80, 100)
    survival_prob = logistic_function(ages, a, b)
    sns.lineplot(x=ages, y=survival_prob, color='red')
    plt.title("Age vs Survival with Logistic Function")
    plt.show()

# Execute script
if __name__ == "__main__":
    main()
