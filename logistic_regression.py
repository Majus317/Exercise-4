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

def bcel(y, y_hat):
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def read_data(path="/Users/hasanatici/Exercise-4/titanic.csv"):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(path)
    
    # Print the column names to ensure correct loading of the CSV
    print("Column Names:", df.columns)  # to help with debug issues
    
    # tend to values
    df.columns = df.columns.str.strip()
    
    # checking if age is empty or not
    if 'Age' not in df.columns:
        print("Error: 'Age' column not found in data!")
    else:
        print("'Age' column is present.")
    
    # Check for missing values and add the mean  to fill in the missing values
    if df['Age'].isnull().sum() > 0:
        print(f"Warning: There are {df['Age'].isnull().sum()} missing values in the 'Age' column.")
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    
    return df


def plot(dataframe):
    plt.figure(figsize=(10, 5))  #
    sns.scatterplot(data=dataframe, x="Age", y="Survived")
    plt.title("Age vs Survival")
    plt.show()

    # Passenger class vs survival scatter plot
    plt.figure(figsize=(10, 5))  
    sns.scatterplot(data=dataframe, x="Pclass", y="Survived")
    plt.title("Passenger Class vs Survival")
    plt.show()


def loss(dataframe, a=0.02, b=0.1):  # Add parameters a and b here
    dataframe["predicted_survival"] = logistic_function(dataframe["Age"], a, b)
    total_loss = bcel(dataframe["Survived"], dataframe["predicted_survival"])
    print(f"Total Binary Cross-Entropy Loss: {total_loss}")
    
    # Plot Age vs Survival with Logistic Regression Line
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=dataframe, x="Age", y="Survived")
    sns.lineplot(x=dataframe["Age"], y=logistic_function(dataframe["Age"], a, b), color="red")
    plt.title("Age vs Survival with Logistic Regression Line")
    plt.show()

# Main execution
if __name__ == "__main__":
    df = read_data("/Users/hasanatici/Exercise-4/titanic.csv")  
    plot(df)
    
    # values for a and b
    loss(df, a=0.02, b=0.1)
