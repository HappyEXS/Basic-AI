import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score

Domian = [-20, 20]

def f(x):
    return ((x ** 2) * np.sin(x)) + (100 * np.sin(x) * np.cos(x))

def plot(x1, y1, title, x2=0, y2=0,):
    plt.plot(x1, y1, 'r')
    plt.plot(x2, y2, 'b')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.title(title)
    plt.savefig(f'{title}.png')
    # plt.show()
    plt.close()

def MLPRegressorModel(layer_size, epochs=5000, mini_batch_size=100, learning_rate=0.001, random_state=3):
    X_train = np.arange(Domian[0], Domian[1], 0.001)
    y_train = f(X_train)
    X_train = X_train.reshape(-1, 1)
    reg = MLPRegressor(hidden_layer_sizes=(layer_size,layer_size),
                        activation='logistic',
                        solver='sgd',
                        batch_size=mini_batch_size,
                        random_state=random_state,
                        verbose=True,
                        learning_rate_init=learning_rate,
                        learning_rate='adaptive',
                        tol=1e-6,
                        max_iter=epochs)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_train)
    plot(X_train, y_train, f"MLPR_N={layer_size}_e={epochs}_rs={random_state}", X_train, y_pred)

def main():
    MLPRegressorModel(30)

if __name__ == "__main__":
    main()