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
   return 1/ (1 + np.exp(-(a*x + b)))
   
# Binary Cross-Entropy loss (logistic loss function)
def bcel(y, y_hat):
    epsilon = 1e-15 # notwendig, damit y_hat zwischen 0 und 1 liegt
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    # Loss function (siehe slides zur 04. Sitzung)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def read_data(path):
    dataframe = pd.read_csv(path, header=0)
    return dataframe


def plot(dataframe, a=1, b=0):
    # 1. The age of a passenger on the x-axis and whether they survived on the y-axis.
    plt.figure(num="First Plot") # Erstellt neues Diagramm und benennt Fenster
    sns.scatterplot(data=dataframe, 
                x="Age",
                y="Survived")
    plt.yticks([0, 1], ["Dead", "Survived"]) # setzt die Werte der y-Achse auf 0 und 1
    plt.title("Age vs. Survived")

    # Logistische Funktion für Age 
    ages = np.linspace(dataframe["Age"].min(), dataframe["Age"].max(), 100)
    survival_prob_age = logistic_function(ages, a, b)
    sns.lineplot(x=ages, y=survival_prob_age, color="red", label=f"logistic(a={a}, b={b})")
    
    # 2. The passenger class on the x-axis and whether they survived on the y-axis.
    plt.figure(num="Second Plot")
    sns.scatterplot(data=dataframe,
                          x="Pclass",
                          y="Survived")
    plt.yticks([0, 1], ["Dead", "Survived"])
    plt.title("Passanger Class vs. Survived")

    # Logistische Funktion für Pclass 
    Pclasses = np.linspace(dataframe["Pclass"].min(), dataframe["Pclass"].max(), 100)
    survival_prob_class = logistic_function(Pclasses, a, b)
    sns.lineplot(x=Pclasses, y=survival_prob_class, color="blue", label=f"logistic(a={a}, b={b})")

    plt.show()

def loss(dataframe, a, b):
    # Berechnung BCE loss für Parameter a und b
    y_age = dataframe["Survived"]
    y_hat_age = logistic_function(dataframe["Age"], a, b)
    age_loss = bcel(y_age, y_hat_age)

    y_class = dataframe["Survived"]
    y_hat_class = logistic_function(dataframe["Pclass"], a, b)
    class_loss = bcel(y_class, y_hat_class)

    print(f"Loss for Age (a={a}, b={b}): {age_loss}")
    print(f"Loss for Class (a={a}, b={b}): {class_loss}")
    return age_loss, class_loss

# Ausführung 
dataframe = read_data("titanic.csv")

# Try different values of a and b to minimize the loss
best_loss = float('inf')
best_params = None
for a in np.linspace(-0.1, 0.1, 5):
    for b in np.linspace(-2, 2, 5):
        age_loss, class_loss = loss(dataframe, a, b)
        avg_loss = (age_loss + class_loss) / 2  # Durchschnittlicher Verlust
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = (a, b)

print(f"Best parameters: a={best_params[0]}, b={best_params[1]} with average loss {best_loss}")

# Besten Parameter plotten: 
plot(dataframe, a=best_params[0], b=best_params[1])
#plot(dataframe)