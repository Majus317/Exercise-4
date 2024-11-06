#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using Titanic data.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def logistic_function(x, a=0, b=0):
    """Logistic function to calculate survival probability."""
    return 1 / (1 + np.exp(-(a * x + b)))

# Binary Cross-Entropy Loss
def bcel(y, y_hat):
    """Calculate binary cross-entropy loss."""
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)  # Avoid log(0)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def read_data(path):
    """Read CSV data file."""
    return pd.read_csv(path)

def plot(dataframe):
    """Plot Age vs. Survival and Class vs. Survival with logistic regression curves."""
    # Plot Age vs. Survival
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='Survived', data=dataframe, alpha=1)
    plt.title("Age vs. Survival")
    plt.xlabel("Age")
    plt.ylabel("Survived")

    # Overlay logistic regression line for Age vs. Survival
    age_values = np.linspace(dataframe['Age'].min(), dataframe['Age'].max(), 100)
    best_a, best_b = find_optimal_params(dataframe['Age'], dataframe['Survived'])
    predicted_probs_age = logistic_function(age_values, best_a, best_b)
    sns.lineplot(x=age_values, y=predicted_probs_age, color='red')
    plt.show()

    # Plot Passenger Class vs. Survival
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Pclass', y='Survived', data=dataframe, alpha=0.6)
    plt.title("Passenger Class vs. Survival")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survived")
    plt.show()

def find_optimal_params(X, y):
    """Find optimal a and b parameters by minimizing the binary cross-entropy loss."""
    best_loss = float("inf")
    best_a, best_b = 0, 0

    a_values = np.linspace(-0.1, 0.1, 50)
    b_values = np.linspace(-10, 10, 50)

    for a in a_values:
        for b in b_values:
            y_pred = logistic_function(X, a, b)
            loss = bcel(y, y_pred)
            if loss < best_loss:
                best_loss = loss
                best_a, best_b = a, b

    return best_a, best_b

def loss(dataframe):
    """Calculate and display binary cross-entropy loss for the best logistic function parameters."""
    # Find optimal parameters
    best_a, best_b = find_optimal_params(dataframe['Age'], dataframe['Survived'])
    
    # Calculate predictions and loss
    y_pred = logistic_function(dataframe['Age'], best_a, best_b)
    calculated_loss = bcel(dataframe['Survived'], y_pred)
    print(f"Optimal parameters: a = {best_a}, b = {best_b}")
    print(f"Minimum binary cross-entropy loss: {calculated_loss}")

# Main execution block
if __name__ == "__main__":
    # Read Titanic data
    df = read_data('titanic.csv')

    # Plot data and logistic regression line
    plot(df)

    # Calculate and print loss
    loss(df)
