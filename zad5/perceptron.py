import numpy as np
from plot import plot
from tqdm import tqdm

Domian = [-5, 5]

def f(x):
    return ((x ** 2) * np.sin(x)) + (100 * np.sin(x) * np.cos(x))

def logistic(x):
    return 1.0/(1 + np.exp(-x))

def logistic_deriv(x):
    return x * (1 - x)

def loss(expected, got):
    return np.square(got-expected)

def loss_derivative(expected, got):
    return (got-expected) * 2

class Network:
    def __init__(self, hidden_layer_size=10):
        # Hidden layer
        self.hidden_layer_weights = np.random.uniform(-1, 1, size=(hidden_layer_size, 1)).astype(np.double)
        self.hidden_layer_biases = np.random.uniform(-1, 1, size=(hidden_layer_size, 1)).astype(np.double)
        # Output layer
        self.output_layer_weights = np.zeros(shape=(1, hidden_layer_size), dtype=np.double)
        self.output_layer_biases = np.zeros(shape=(1, 1), dtype=np.double)

    def predict(self, x):
        prediction = np.array([x], dtype=np.double)
        # Hidden layer
        prediction = logistic((self.hidden_layer_weights @ prediction) + self.hidden_layer_biases)
        # Output layer
        prediction = (self.output_layer_weights @ prediction) + self.output_layer_biases
        return prediction[0][0]

    def fit(self, train_inputs, train_outputs, epochs, mini_batch_size=100, learning_rate=0.01):
        train_indices = np.arange(len(train_inputs), dtype=np.intc)
        for i in tqdm(range(epochs)):
            self.batch(train_inputs, train_outputs, train_indices, mini_batch_size, learning_rate)

    def batch(self, train_inputs, train_outputs, train_indices, mini_batch_size, learning_rate):
        np.random.shuffle(train_indices)                        # Shuffling train indices to randomize data in mini_batches
        mini_batches = len(train_indices) // mini_batch_size    # Calculating number of mini_batches
        for i in range(mini_batches):
            # Start & end of data in mini_batch
            start = mini_batch_size * i
            end = start + mini_batch_size

            self.mini_batch(train_inputs, train_outputs, train_indices[start:end], learning_rate)

    def mini_batch(self, train_inputs, train_outputs, train_indices,  learning_rate):
        # Declaration of gradients with zeros
        grad_hidden_weights = np.zeros(shape=self.hidden_layer_weights.shape, dtype=np.double)
        grad_hidden_biases = np.zeros(shape=self.hidden_layer_biases.shape, dtype=np.double)
        grad_output_weights = np.zeros(shape=self.output_layer_weights.shape, dtype=np.double)
        grad_output_biases = np.zeros(shape=self.output_layer_biases.shape, dtype=np.double)

        # For every training data in mini_batch adding backpropagation gradient values
        for i in train_indices:
            delta_hw, delta_hb, delta_ow, delta_ob = self.backprop(train_inputs[i], train_outputs[i])
            grad_hidden_weights += delta_hw
            grad_hidden_biases += delta_hb
            grad_output_weights += delta_ow
            grad_output_biases += delta_ob

        # Multiplaying gradients by learning rate parameter
        mult = learning_rate / len(train_indices)
        grad_hidden_weights *= mult
        grad_hidden_biases *= mult
        grad_output_weights *= mult
        grad_output_biases *= mult

        # Adding the opposite values of gradnients to hidden and output layers weights and biases
        self.hidden_layer_weights -= grad_hidden_weights
        self.hidden_layer_biases -= grad_hidden_biases
        self.output_layer_weights -= grad_output_weights
        self.output_layer_biases -= grad_output_biases

    def backprop(self, input, expected_output):
        # Forward propagation
        activations_output_expected = np.array([[expected_output]], dtype=np.double)
        activations_input = np.array([[input]], dtype=np.double)

        activations_hidden = logistic((self.hidden_layer_weights @ activations_input) + self.hidden_layer_biases)
        activations_output = (self.output_layer_weights @ activations_hidden) + self.output_layer_biases

        # Backwards propagation into the output layer
        delta_ob = loss_derivative(activations_output_expected, activations_output)
        delta_ow = delta_ob @ activations_hidden.T

        # Backpropagate into the hidden layer
        delta_hb = self.output_layer_weights.T @ delta_ob
        delta_hb *= logistic_deriv(activations_hidden)
        delta_hw = delta_hb @ activations_input

        return delta_hw, delta_hb, delta_ow, delta_ob

def experiment(neurons, epochs, mini_batch, learning_rate):
    X_train = np.linspace(Domian[0], Domian[1], 10_000, dtype=np.double)
    y_train = f(X_train)
    model = Network(neurons)
    model.fit(X_train, y_train, epochs, mini_batch, learning_rate)
    y_pred = [model.predict(x) for x in X_train]

    plot(X_train, y_train, f"n={neurons}; ep={epochs}; mb={mini_batch}; lr={learning_rate}_(-5;5)", X_train, y_pred)

def main():
    # N = [5, 10, 15, 20, 25]
    # N = [8, 10, 12, 14, 16]
    # for n in N:
    #     experiment(neurons=n, epochs=1000, mini_batch=100, learning_rate=0.01)
    experiment(neurons=10, epochs=200, mini_batch=100, learning_rate=0.01)

if __name__ == "__main__":
    main()