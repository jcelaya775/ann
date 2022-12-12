import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from joblib import Memory


def get_mnist(batch_size=64, random_seed=42):
    def split_into_batches(x, batch_size):
        n_batches = len(x) / batch_size
        x = np.array_split(x, n_batches)
        return np.array(x, dtype=object)

    # Cache the downloaded dataset
    memory = Memory("./mnist")
    fetch_openml_cached = memory.cache(fetch_openml)
    mnist = fetch_openml_cached("mnist_784")

    # Split data into training and validation sets
    x_train, x_test, y_train, y_test = train_test_split(
        mnist.data, mnist.target, test_size=0.33, random_state=random_seed
    )

    # Normalize the data
    min_max_scalar = MinMaxScaler()

    # One-Hot endcode the targets
    one_hot_encoder = OneHotEncoder()

    # Split the training data into batches
    x_train = split_into_batches(
        min_max_scalar.fit_transform(np.array(x_train)), batch_size
    )

    # Split the tesing data into batches
    x_test = split_into_batches(
        min_max_scalar.fit_transform(np.array(x_test)), batch_size
    )

    # Turn the targets into NumPy arrays and flatten the array
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    # One-Hot endcode the training and testing data and split it into batches
    one_hot_encoder.fit(y_train)
    y_train = one_hot_encoder.transform(y_train).toarray()
    y_train = split_into_batches(np.array(y_train), batch_size)

    one_hot_encoder.fit(y_test)
    y_test = one_hot_encoder.transform(y_test).toarray()
    y_test = split_into_batches(np.array(y_test), batch_size)

    return x_train, y_train, x_test, y_test


# Initializes the weights and biases randomly using the random normal distribution
def init_weights_and_biases(nn_architecture):
    parameters = {}
    for idx, layer in enumerate(nn_architecture):
        n_input = layer["in"]
        n_output = layer["out"]

        parameters[f"weights {idx}->{idx+1}"] = np.random.randn(n_input, n_output) * 0.1
        parameters[f"bias {idx+1}"] = (np.random.randn(n_output,) * 0.1) # notice output is initialized with a comma
    return parameters


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def forward_single_layer(a_prev, w, b, activation_function):
    z = a_prev @ w + b
    a = activation_function(z)
    return a, z


def forward(x, nn_parameters, nn_architecture):
    # Our network consists of 3 layers, but our dictionary contains only input and output layers
    n_layers = len(nn_architecture) + 1 # so let's add 1
    # Memory is needed later for backpropagation
    memory = {}
    a = x

    for i in range(1, n_layers): # skip input layer
        a_prev = a
        activation_function = globals()[nn_architecture[i - 1]["activation"]]
        w = nn_parameters[f"weights {i-1}->{i}"]
        b = nn_parameters[f"bias {i}"]

        a, z = forward_single_layer(a_prev, w, b, activation_function)

        memory[f"a_{i - 1}"] = a_prev
        memory[f"z_{i}"] = z

    return a, memory


def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

    
def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def backpropagation_single_layer(dA, w, z, a_prev, activation_function):
    m = a_prev.shape[0]
    backprop_activation = globals()[f"{activation_function}_backward"]

    delta = backprop_activation(dA, z)
    dW = (a_prev.T @ delta) / m
    dB = np.sum(delta, axis=1) / m
    dA_prev = delta @ w.T

    return dW, dB, dA_prev


def backward(target, prediction, memory, param_values, nn_architecture):
    gradients = {}
    dA_prev = 2 * (prediction - target)
    n_layers = len(nn_architecture)  + 1
    
    # Loop backwards
    for i in reversed(range(1, n_layers)):
        dA = dA_prev

        # Memory from the forward propogation step
        a_prev = memory[f"a_{i - 1}"]
        z = memory[f"z_{i}"]

        w = param_values[f"weights {i - 1}->{i}"]

        dW, dB, dA_prev = backpropagation_single_layer(
            dA, w, z, a_prev, nn_architecture[i - 1]["activation"]
        )

        gradients[f"dW_{i - 1}->{i}"] = dW
        gradients[f"dB_{i}"] = dB
    
    return gradients

def update(param_values, gradients, nn_architecture, learning_rate):
    n_layers = len(nn_architecture) + 1
    for i in range(1, n_layers):
        param_values[f"weights {i - 1}->{i}"] -= (learning_rate * gradients[f"dW_{i - 1}->{i}"])
        param_values[f"bias {i}"] -= learning_rate * gradients[f"dB_{i}"].mean()
    return param_values


def get_current_accuracy(param_values, nn_architecture, x_test, y_test):
    correct = 0
    total_counter = 0
    for x, y in zip(x_test, y_test):
        for i in range(len(x)):
            a, _ = forward(x, param_values, nn_architecture)
            pred = np.argmax(a, axis=1)
            y = np.argmax(y, axis=1)

            plt.imshow(x[i].reshape(28, 28))
            plt.title(f"It's a {y[i]}. The NN thinks it's a {pred[i]}!")
            plt.show()

            correct += (pred == y).sum()
            total_counter += len(x)
    accuracy = correct / total_counter
    return accuracy


def main():
    # m = 784 samples, n = 16 features, o = 10 classes (for indentifying numbers 1-10)
    neural_network = [
        {"in": 784, "out": 16, "activation": "relu"},
        {"in": 16, "out": 10, "activation": "relu"},
    ]
    
    # Train the model
    x_train, y_train, x_test, y_test = get_mnist()
    parameters = init_weights_and_biases(neural_network)
    n_epochs = 75
    learning_rate = 0.1
    for epoch in range(n_epochs):
        for x, y in zip(x_train, y_train):
            a, memory = forward(x, parameters, neural_network)
            grads = backward(y, a, memory, parameters, neural_network)
            update(parameters, grads, neural_network, learning_rate)
        accuracy = get_current_accuracy(parameters, neural_network, x_test, y_test)
        print(f"Epoch { epoch } Accuracy = { np.round(accuracy * 100, 4) }")


main()