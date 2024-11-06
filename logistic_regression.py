#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

titanic_data = pd.read_csv('titanic.csv')

# Plot 1
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Survived', data=titanic_data, alpha=0.6)
plt.title('Age vs Survival')
plt.xlabel('Age')
plt.ylabel('Survived')
plt.show()

# Plot 2
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Pclass', y='Survived', data=titanic_data, alpha=0.6)
plt.title('Passenger Class vs Survival')
plt.xlabel('Passenger Class')
plt.ylabel('Survived')
plt.show()

# Logistic function
def logistic(x, a, b):
    return 1 / (1 + np.exp(-(a * x + b)))

# Binary cross-entropy loss function
def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) 
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


ages = titanic_data['Age'].dropna()
survived = titanic_data['Survived'].dropna()

a, b = 0.001, -0.4  # values

age_range = np.linspace(ages.min(), ages.max(), 100)
logistic_predictions = logistic(age_range, a, b)
loss = binary_cross_entropy(survived, logistic(ages, a, b))

# Plot Age v Survival 
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Survived', data=titanic_data, alpha=0.6)
sns.lineplot(x=age_range, y=logistic_predictions, color='red')
plt.title(f'Age vs Survival (Logistic Regression Line) - Loss: {loss:.4f}')
plt.xlabel('Age')
plt.ylabel('Survived')
plt.show()

# Logistic Regression Class v Survival
pclass = titanic_data['Pclass']

a_pclass, b_pclass = -0.7, 0.9  #values
pclass_range = np.linspace(pclass.min(), pclass.max(), 100)
logistic_predictions_pclass = logistic(pclass_range, a_pclass, b_pclass)
loss_pclass = binary_cross_entropy(survived, logistic(pclass, a_pclass, b_pclass))

# Plot Class v Survival
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Pclass', y='Survived', data=titanic_data, alpha=0.6)
sns.lineplot(x=pclass_range, y=logistic_predictions_pclass, color='blue')
plt.title(f'Passenger Class vs Survival (Logistic Regression Line) - Loss: {loss_pclass:.4f}')
plt.xlabel('Passenger Class')
plt.ylabel('Survived')
plt.show()