#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# CSV-Datei laden
df = pd.read_csv("titanic.csv")

# Scatterplot: Alter vs Überleben
sns.scatterplot(x='Age', y='Survived', data=df)
plt.title("Age vs Survival")
plt.show()

# Scatterplot: Klasse vs Überleben
sns.scatterplot(x='Pclass', y='Survived', data=df)
plt.title("Class vs Survival")
plt.show()



import numpy as np

# Logistische Funktion
def logistic(x, a, b):
    return 1 / (1 + np.exp(-(a * x + b)))

# Cross-Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


a, b = 2, -5

# Alter vs Überleben mit logistischem Fit
df['Predicted_Survival'] = logistic(df['Age'], a, b)
sns.scatterplot(x='Age', y='Survived', data=df)
sns.lineplot(x='Age', y='Predicted_Survival', data=df, color="red")
plt.title("Age vs Survival with Logistic Fit")
plt.show()

# berechnet und gib  Loss aus
loss = binary_cross_entropy(df['Survived'], df['Predicted_Survival'])
print("Binary Cross-Entropy Loss:", loss)


