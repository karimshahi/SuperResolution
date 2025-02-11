import numpy as np

class RLFN:
    def __init__(self, input_dim, output_dim, num_features):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_features = num_features
        self.weights = np.random.rand(num_features, input_dim)
        self.biases = np.random.rand(num_features)

    def forward(self, x):
        # Calculate the random linear features
        features = np.dot(self.weights, x) + self.biases
        # Apply the activation function (e.g., ReLU)
        features = np.maximum(features, 0)
        # Calculate the output
        output = np.dot(features, np.random.rand(self.num_features, self.output_dim))
        return output

    def train(self, x_train, y_train):
        # Calculate the random linear features
        features = np.dot(self.weights, x_train.T) + self.biases[:, np.newaxis]
        # Apply the activation function (e.g., ReLU)
        features = np.maximum(features, 0)
        # Calculate the output weights
        output_weights = np.dot(np.linalg.pinv(features), y_train.T)
        # Update the output weights
        self.output_weights = output_weights

    def predict(self, x_test):
        # Calculate the random linear features
        features = np.dot(self.weights, x_test.T) + self.biases[:, np.newaxis]
        # Apply the activation function (e.g., ReLU)
        features = np.maximum(features, 0)
        # Calculate the output
        output = np.dot(features.T, self.output_weights)
        return output

# Example usage:
np.random.seed(42)
input_dim = 10
output_dim = 2
num_features = 50

x_train = np.random.rand(100, input_dim)
y_train = np.random.rand(100, output_dim)

rlfn = RLFN(input_dim, output_dim, num_features)
rlfn.train(x_train, y_train)

x_test = np.random.rand(20, input_dim)
y_pred = rlfm.predict(x_test)
print(y_pred)
