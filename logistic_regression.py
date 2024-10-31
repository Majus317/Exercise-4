#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic_df = pd.read_csv('titanic.csv')

def logistic_function(x, a=0, b=0):
    return 1 / (1 + np.exp(-(a * x + b)))

def bcel(y, y_hat):
    return -np.mean(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))


a_age, b_age = 0.05, -1  # for age
a_pclass, b_pclass = -1, 3  # for passenger class

titanic_df['pred_survived_age'] = logistic_function(titanic_df['Age'].fillna(titanic_df['Age'].mean()), a_age, b_age)
titanic_df['pred_survived_pclass'] = logistic_function(titanic_df['Pclass'], a_pclass, b_pclass)


age_loss = bcel(titanic_df['Survived'], titanic_df['pred_survived_age'])
pclass_loss = bcel(titanic_df['Survived'], titanic_df['pred_survived_pclass'])

print(f"Initial Loss (Age): {age_loss:.4f}")
print(f"Initial Loss (Pclass): {pclass_loss:.4f}")


plt.figure(figsize=(10, 5))
sns.scatterplot(x='Age', y='Survived', data=titanic_df, alpha=0.5, label="Observed")
sns.lineplot(x='Age', y='pred_survived_age', data=titanic_df, color="red", label="Logistic Regression")
plt.title('Age vs Survival with Logistic Regression')
plt.xlabel('Age')
plt.ylabel('Survived')
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))
sns.scatterplot(x='Pclass', y='Survived', data=titanic_df, alpha=0.5, label="Observed")
sns.lineplot(x='Pclass', y='pred_survived_pclass', data=titanic_df, color="red", label="Logistic Regression")
plt.title('Passenger Class vs Survival with Logistic Regression')
plt.xlabel('Passenger Class')
plt.ylabel('Survived')
plt.legend()
plt.show()


