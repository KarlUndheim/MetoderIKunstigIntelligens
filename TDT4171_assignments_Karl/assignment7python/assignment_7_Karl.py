import numpy as np


def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    # TODO: Your code goes here.

    # After some experimentation I found 0.5 to use well as learning rate.
    learning_rate = 5e-1

    # Initialize weights and biases
    # This list will hold two arrays, each array contains the weights of its layer
    weights = []
    w1_shape = (2, 2)
    w2_shape = (2, 1)
    # I chose a uniform distribution centered around 0. This usually works well for initializing weights.
    w1 = np.random.uniform(-1, 1, w1_shape)
    w2 = np.random.uniform(-1, 1, w2_shape)
    weights.append(w1)
    weights.append(w2)
    
    # Since the gradients are different for the biases, I keep them in a separate list.
    biases = []
    b1_shape = (1,2)
    b2_shape = (1,1)
    b1 = np.random.uniform(-1, 1, b1_shape)
    b2 = np.random.uniform(-1, 1, b2_shape)
    biases.append(b1)
    biases.append(b2)

    # These lists will be identical in shape as the weights and biases lists. This way updating
    # the weights and the biases later will be easy. 
    gradients = [None for i in range(len(weights))]
    biasGradients = [None for i in range(len(biases))]

    # I will store the activations of the different layers including the first layer as itself.
    activations = [None for i in range(3)]

    # Here I calculate the error before any training is done so that I can compare with the trained model later.
    # I will explain the forward pass afterwards in the trainin loop.
    mse = 0
    for X, target in zip(X_test, y_test):
        activations[0] = X.reshape((1,2))

        # z = a*w + b
        z_0 = np.dot(activations[0], weights[0])+biases[0]

        # Pass z through the sigmoid function and we get the activation for the next layer
        activations[1] = 1/(1+np.exp(-z_0))

        # z = a*w + b
        z_1 = np.dot(activations[1], weights[1])+biases[1]

        # Linear activation
        activations[2] = z_1

        mse+=(z_1-target)**2
    mse/=len(X_test)
    print("Mean square error before training: ", mse)


    # TRAINING LOOP
    # For each training example I will perform:
    # 1. Forward pass
    # 2. Backpropagation
    # 3. Update weights and biases
    for X, target in zip(X_train, y_train):

        # 1. Forward pass

        # Set first activation layer as the input, this will be used when calculating the activations of the next layer.
        activations[0] = X.reshape((1,2))

        # z = a*w + b
        z_0 = np.dot(activations[0], weights[0])+biases[0]

        # Pass z through the sigmoid function and we get the activation for the next layer
        activations[1] = 1/(1+np.exp(-z_0))

        # z = a*w + b
        z_1 = np.dot(activations[1], weights[1])+biases[1]

        # Linear activation
        activations[2] = z_1



        # 2. Backpropagation

        output = activations[2]

        # Delta of final layer. Since we use a linear activation for the final layer, the derivative is 1 and the delta becomes:
        delta = output-target
        
        # The update rule for the gradients is simpy the error of its node.
        biasGradients[1] = delta

        # Update rule for weight gradients:
        gradients[1] = delta*activations[1].T

        # Here I propagate the error(delta) backwards
        derivative = activations[1] * (1-activations[1])
        delta = (derivative*np.dot(delta, weights[1].T)).T

        biasGradients[0] = delta.T
        gradients[0] = np.dot(delta, activations[0]).T

        # For each layer: update the weights and biases of the layer
        for i in range(len(weights)):
            weights[i] = weights[i] - learning_rate*gradients[i]
            biases[i] = biases[i] - learning_rate*biasGradients[i]

    # This rest is for calculating mse
    mse = 0
    for X, target in zip(X_test, y_test):
        activations[0] = X.reshape((1,2))

        # z = a*w + b
        z_0 = np.dot(activations[0], weights[0])+biases[0]

        # Pass z through the sigmoid function and we get the activation for the next layer
        activations[1] = 1/(1+np.exp(-z_0))

        # z = a*w + b
        z_1 = np.dot(activations[1], weights[1])+biases[1]

        # Linear activation
        activations[2] = z_1

        mse+=(z_1-target)**2
    mse/=len(X_test)
    print("Mean square error after training: ", mse)

    mse = 0
    for X, target in zip(X_train, y_train):
        activations[0] = X.reshape((1,2))

        # z = a*w + b
        z_0 = np.dot(activations[0], weights[0])+biases[0]

        # Pass z through the sigmoid function and we get the activation for the next layer
        activations[1] = 1/(1+np.exp(-z_0))

        # z = a*w + b
        z_1 = np.dot(activations[1], weights[1])+biases[1]

        # Linear activation
        activations[2] = z_1

        mse+=(z_1-target)**2
    mse/=len(X_train)
    print("Mean square error TRAINING SET: ", mse)
    
