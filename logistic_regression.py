#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Logistic function
def logistic_function(x, a=1, b=0):
    """Computes the logistic function with parameters a and b."""
    return 1 / (1 + np.exp(-(a * x + b)))

# Binary Cross-Entropy Loss
def bcel(y, y_hat):
    """Calculates binary cross-entropy loss between actual and predicted values."""
    epsilon = 1e-15  # To avoid log(0)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)  # Keep predictions within bounds
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# Read data from CSV file
def read_data(path):
    """Reads Titanic data from CSV and preprocesses it for analysis."""
    df = pd.read_csv(path)
    df = df[['Age', 'Pclass', 'Survived']].dropna()  # Select relevant columns and drop missing values
    return df

# Plotting function
def plot(dataframe):
    """Generates scatter plots for Age vs. Survived and Pclass vs. Survived."""
    plt.figure(figsize=(12, 5))
    
    # Age vs. Survived
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Age', y='Survived', data=dataframe, alpha=0.5)
    plt.title('Age vs Survived')
    plt.xlabel('Age')
    plt.ylabel('Survived')
    
    # Pclass vs. Survived
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='Pclass', y='Survived', data=dataframe, alpha=0.5)
    plt.title('Pclass vs Survived')
    plt.xlabel('Pclass')
    plt.ylabel('Survived')
    
    plt.tight_layout()
    plt.show()

# Loss calculation and logistic function line plot
def loss(dataframe, a=1, b=0):
    """Calculates and prints loss for Age and Pclass, and overlays logistic regression lines."""
    # Age vs. Survived with logistic curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Age', y='Survived', data=dataframe, alpha=0.5)
    age_vals = np.linspace(dataframe['Age'].min(), dataframe['Age'].max(), 100)
    survival_probs_age = logistic_function(age_vals, a, b)
    sns.lineplot(x=age_vals, y=survival_probs_age, color='red')
    plt.title(f'Age vs Survived with Logistic Curve (a={a}, b={b})')
    plt.xlabel('Age')
    plt.ylabel('Survived')
    
    # Calculate binary cross-entropy loss for Age
    age_preds = logistic_function(dataframe['Age'], a, b)
    loss_age = bcel(dataframe['Survived'], age_preds)
    print(f'Binary Cross-Entropy Loss for Age vs Survived: {loss_age}')
    
    # Pclass vs. Survived with logistic curve
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='Pclass', y='Survived', data=dataframe, alpha=0.5)
    pclass_vals = np.linspace(dataframe['Pclass'].min(), dataframe['Pclass'].max(), 100)
    survival_probs_pclass = logistic_function(pclass_vals, a, b)
    sns.lineplot(x=pclass_vals, y=survival_probs_pclass, color='red')
    plt.title(f'Pclass vs Survived with Logistic Curve (a={a}, b={b})')
    plt.xlabel('Pclass')
    plt.ylabel('Survived')
    
    # Calculate binary cross-entropy loss for Pclass
    pclass_preds = logistic_function(dataframe['Pclass'], a, b)
    loss_pclass = bcel(dataframe['Survived'], pclass_preds)
    print(f'Binary Cross-Entropy Loss for Pclass vs Survived: {loss_pclass}')
    
    plt.tight_layout()
    plt.show()

# Main function to run the steps
def main():
    path = 'titanic.csv'  # Replace with your file path if needed
    df = read_data(path)
    plot(df)
    # Experiment with different values of `a` and `b` to minimize the loss
    loss(df, a=0.05, b=-3)  # Example values, adjust to find the best fit

# Run the main function
if __name__ == "__main__":
    main()
