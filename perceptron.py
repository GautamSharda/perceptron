import os
import csv
import matplotlib.pyplot as plt

# Path to directory that stores results of training runs
TRAINING_RUN_DATA_DIRECTORY = '/training_runs'

class perceptron:
    def __init__(self, weights = [50, -50], learning_rate = 0.01):
        # Initialize weights randomly between -50 and 50
        self.weights = weights
        self.learning_rate = learning_rate

    def predict(self, x1, x2):
        """Function to predict the output using the current weights."""
        return self.weights[0]*x1 + self.weights[1]*x2

    def compute_error(self, actual, predicted):
        """Compute the squared error."""
        return (actual - predicted) ** 2

    def update_weights(self, x1, x2, actual):
        """Update weights using the partial derivative of the error function w.r.t each weight."""
        predicted = self.predict(x1, x2)
        error = actual - predicted
        
        # Derivatives of error function w.r.t. weights
        dEdW1 = -2 * x1 * error
        dEdW2 = -2 * x2 * error
        
        # Update weights
        self.weights[0] -= self.learning_rate * dEdW1
        self.weights[1] -= self.learning_rate * dEdW2

    def load_training_data(self, file_name, batch_size):
        TRAINING_DATA = []
        TARGETS = []
        # Read the CSV file
        with open(file_name, mode='r', newline='') as file:
            reader = csv.reader(file)
            
            # Iterate over the rows in the file
            for i, row in enumerate(reader):
                # Break the loop after reading 9000 rows
                if i >= batch_size:
                    break
                TRAINING_DATA.append([float(row[0]), float(row[1])])
                TARGETS.append(float(row[2]))
        return TRAINING_DATA, TARGETS

    def train(self, data_file, epochs, batch_size, results_file):
        TRAINING_DATA, TARGETS = self.load_training_data(data_file, batch_size)
        # Training loop
        print("Initial weights:", self.weights)
        errors = []
        for epoch in range(epochs):
            for x1, x2, actual in zip(*zip(*TRAINING_DATA), TARGETS):
                self.update_weights(x1, x2, actual)
                predicted = self.predict(x1, x2)
                errors.append(self.compute_error(actual, predicted))

            print(f"Epoch {epoch + 1}, Weights: {self.weights}")

        # Plotting the error over iterations
        plt.figure(figsize=(10, 6))
        plt.plot(errors)
        plt.title('Loss Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()
        
        print(self.predict(1, 2))

        # construct result data
        # store_training_data(results_file, data)
        
    
    def store_training_data(self, file_name, data):
        # Data to be written to the CSV file
        data = [
            ["Name", "Age", "City"],  # Header row
            ["Alice", 28, "New York"],
            ["Bob", 22, "San Francisco"],
            ["Charlie", 33, "Los Angeles"]
        ]

        file_path = os.path.join(TRAINING_RUN_DATA_DIRECTORY, f'{file_name}.csv')

        # Open a new CSV file for writing
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the data to the CSV file
            for row in data:
                writer.writerow(row)

learner = perceptron()
learner.train(data_file='addition_floats.csv', epochs=1, batch_size=9000, results_file=None)


'''
import csv
import random
import matplotlib.pyplot as plt

weights = [random.random() for _ in range(2)]
learning_rate = 0.01 

def approximate_partial_derivative(f, g, weights, i, actual):
    h = 0.001
    temp_weights = weights.copy()  # Work with a copy to avoid changing original weights
    
    # Calculate the change in the weights for the ith weight
    temp_weights[i] += h
    
    # Approximate the derivative of the outer function (error) at g(x, y)
    outer_derivative_approx = (f(g(*temp_weights), actual) - f(g(*weights), actual)) / h
    
    # Approximate the partial derivative of the inner function (weighted sum) with respect to x at (x, y)
    inner_derivative_approx = (g(*temp_weights) - g(*weights)) / h
    
    # Apply the chain rule: multiply the outer derivative by the inner derivative
    return outer_derivative_approx * inner_derivative_approx

def weighted_sum(x1, x2):
    input = [x1, x2]
    value = 0
    for i in range(len(weights)):
        value += weights[i]*input[i]
    return value

def activation_function(value):
    #ignore for now, since i guess it only enables non-linear approximation, which for now we will ignore
    if value < 0:
        return 0
    else:
        return 1

def _error(actual, estimate):
    return (estimate - actual)**2

def update_weights(sample):
    weight_gradient = []
    for i in range(len(weights)):
        # let's start with stochastic gradient descent
        # take partial derivative of error function wrt each weight
        approx_derivative_of_error_wrt_weight_i = approximate_partial_derivative(_error, weighted_sum, weights, i, int(sample[2]))
        print(f"The approximate partial derivative of the composite function with respect to weight {i} is: {approx_derivative_of_error_wrt_weight_i}")
        weight_gradient.append(approx_derivative_of_error_wrt_weight_i)
    for i in range(len(weights)):
        weights[i] = weights[i] - learning_rate*weight_gradient[i]
'''