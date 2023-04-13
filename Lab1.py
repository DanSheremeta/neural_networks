import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert y values to 1 or -1
        y_ = np.array([1 if i > 0 else -1 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(x):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Update weights and bias
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, -1)


# Generate random training data
X_train = np.random.uniform(low=-2, high=2, size=(100, 3))
Y_train = np.where(X_train.sum(axis=1) > 0, 1, -1)

# Create the model
model = Perceptron(learning_rate=0.1, n_iters=100)

# Train the model
model.fit(X_train, Y_train)

# Plot the decision boundary
x1 = np.linspace(-2, 2, 100)
x2 = -1 * (model.weights[0] * x1 + model.bias) / model.weights[1]
plt.plot(x1, x2, '-r', label='Perceptron')
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train)
plt.legend()
plt.show()
