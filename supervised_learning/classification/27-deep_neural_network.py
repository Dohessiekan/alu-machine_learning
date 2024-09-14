#!/usr/bin/env python3
"""
Defines the DeepNeuralNetwork class that implements
a deep neural network for multiclass classification.
"""
import numpy as np

class DeepNeuralNetwork:
    """
    Deep neural network performing multiclass classification.
    """

    def __init__(self, nx, layers):
        """
        Initializes the network with the given number of input features and layers.

        Parameters:
            nx (int): The number of input features.
            layers (list of int): Number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer or layers are not a list of positive integers.
            ValueError: If nx is less than 1.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) < 1 or not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        previous = nx
        for i, layer in enumerate(layers):
            self.__weights["W{}".format(i + 1)] = np.random.randn(layer, previous) * np.sqrt(2 / previous)
            self.__weights["b{}".format(i + 1)] = np.zeros((layer, 1))
            previous = layer

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
    def forward_prop(self, X):
        """
        Performs forward propagation for the neural network.

        Parameters:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            The final layer's output and the cache.
        """
        self.__cache["A0"] = X
        for i in range(self.L):
            W = self.__weights["W{}".format(i + 1)]
            b = self.__weights["b{}".format(i + 1)]
            Z = np.matmul(W, self.__cache["A{}".format(i)]) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache["A{}".format(i + 1)] = A
        return A, self.__cache
    def cost(self, Y, A):
        """
        Computes the cost using logistic regression.

        Parameters:
            Y (numpy.ndarray): True labels (one-hot) with shape (classes, m).
            A (numpy.ndarray): Predicted output with shape (classes, m).

        Returns:
            The cost (float).
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        Parameters:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): True labels.

        Returns:
            The prediction and the cost.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
    def gradient_descent(self, Y, alpha=0.05):
        """
        Performs one pass of gradient descent on the neural network.

        Parameters:
            Y (numpy.ndarray): True labels.
            alpha (float): Learning rate.
        """
        m = Y.shape[1]
        L = self.L
        dz = self.__cache["A{}".format(L)] - Y
        for i in reversed(range(1, L + 1)):
            A_prev = self.__cache["A{}".format(i - 1)]
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            dW = (1 / m) * np.matmul(dz, A_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            if i > 1:
                A_prev = self.__cache["A{}".format(i - 1)]
                dz = np.matmul(W.T, dz) * A_prev * (1 - A_prev)
            self.__weights["W{}".format(i)] -= alpha * dW
            self.__weights["b{}".format(i)] -= alpha * db
    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neural network.

        Parameters:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): True labels.
            iterations (int): Number of iterations to train.
            alpha (float): Learning rate.
            verbose (bool): If True, print progress during training.
            graph (bool): If True, plot cost function over time.
            step (int): How often to print progress and plot.

        Returns:
            The evaluation after training.
        """
        if type(iterations) is not int or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float or alpha <= 0:
            raise ValueError("alpha must be positive")
        costs = []
        for i in range(iterations):
            A, _ = self.forward_prop(X)
            self.gradient_descent(Y, alpha)
            if verbose and i % step == 0:
                cost = self.cost(Y, A)
                print(f"Cost after {i} iterations: {cost}")
                costs.append(cost)
        if graph:
            import matplotlib.pyplot as plt
            plt.plot(np.arange(0, iterations, step), costs)
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

