#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

# Logistische Funktion
def logistic(x, a, b):
    return 1 / (1 + np.exp(-(a * x + b)))

# Cross-Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    # Vermeidung von log(0)-Fehlern, indem y_pred leicht eingeschränkt wird
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# für  logistische Funktion
a, b = 1, -3  

# Alter vs Überleben mit logistischem Fit
df = df.dropna(subset=['Age']) 
df['Predicted_Survival'] = logistic(df['Age'], a, b)
sns.scatterplot(x='Age', y='Survived', data=df)
sns.lineplot(x='Age', y='Predicted_Survival', data=df, color="red")
plt.title("Age vs Survival with Logistic Fit")
plt.show()

# berechnet und gib Binary Cross-Entropy Loss aus
loss = binary_cross_entropy(df['Survived'], df['Predicted_Survival'])
print("Binary Cross-Entropy Loss:", loss)
