import pandas
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

D = [-20, 20]

def f(x):
    return ((x ** 2) * np.sin(x)) + (100 * np.sin(x) * np.cos(x))

def plot(x1, y1, title, x2=0, y2=0,):
    plt.plot(x1, y1, 'r')
    plt.plot(x2, y2, 'b')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.title(title)
    # plt.savefig(f'{title}.png')
    plt.show()
    plt.close()

def logistic(x):
    return 1.0/(1 + np.exp(-x))

def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))

def myModel(x_train, x_test, y_train, y_test):
    LR = 1

    I_dim = 3
    H_dim = 4
    epoch_count = 1

    weights_ItoH = np.random.uniform(-1, 1, (I_dim, H_dim))
    weights_HtoO = np.random.uniform(-1, 1, H_dim)

    preActivation_H = np.zeros(H_dim)
    postActivation_H = np.zeros(H_dim)

    for epoch in range(epoch_count):
        # Learn each epoch
        for sample in range(len(x_train)):
            for node in range(H_dim):
                preActivation_H[node] = np.dot(x_train[sample:], weights_ItoH[:node])
                postActivation_H[node] = logistic(preActivation_H[node])

            preActivation_O = np.dot(postActivation_H, weights_HtoO)
            postActivation_O = logistic(preActivation_O)

            FE = postActivation_O - x_test[sample]

            for H_node in range(H_dim):
                S_error = FE * logistic_deriv(preActivation_O)
                gradient_HtoO = S_error * postActivation_H[H_node]

                for I_node in range(I_dim):
                    input_value = training_data[sample, I_node]
                    gradient_ItoH = S_error * weights_HtoO[H_node] * logistic_deriv(preActivation_H[H_node]) * input_value

                    weights_ItoH[I_node, H_node] -= LR * gradient_ItoH

                weights_HtoO[H_node] -= LR * gradient_HtoO

        # Test model after each epoch
        correct_classification_count = 0
        for sample in range(len(x_test)):
            for node in range(H_dim):
                preActivation_H[node] = np.dot(x_test[sample,:], weights_ItoH[:, node])
                postActivation_H[node] = logistic(preActivation_H[node])

            preActivation_O = np.dot(postActivation_H, weights_HtoO)
            postActivation_O = logistic(preActivation_O)

            if postActivation_O > 0.5:
                output = 1
            else:
                output = 0

            if output == y_test[sample]:
                correct_classification_count += 1

        print(f'Epoch: {epoch}')
        print("Accuracy: ",correct_classification_count*100/validation_count)




def main():
    X = np.arange(D[0], D[1], 0.1)
    y = f(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    myModel(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()