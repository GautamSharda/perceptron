# practicing writing a perceptron to estimate the AND function: 
# f(x, y) = {  1, x >= 1, y >= 1
#              0, otherwise       }

import csv
import matplotlib.pyplot as plt

# Initialize weights randomly between -50 and 50
weights = [50, -50] 
print("Initial weights:", weights)
learning_rate = 0.01

def predict(x1, x2):
    """Function to predict the output using the current weights."""
    return weights[0]*x1 + weights[1]*x2

def compute_error(actual, predicted):
    """Compute the squared error."""
    return (actual - predicted) ** 2

def update_weights(x1, x2, actual):
    """Update weights using the partial derivative of the error function w.r.t each weight."""
    predicted = predict(x1, x2)
    error = actual - predicted
    
    # Derivatives of error function w.r.t. weights
    dEdW1 = -2 * x1 * error
    dEdW2 = -2 * x2 * error
    
    # Update weights
    weights[0] -= learning_rate * dEdW1
    weights[1] -= learning_rate * dEdW2

# Dummy data for training (you would replace this with your actual data)
data = []
targets = []

# Define the file name and the maximum number of rows to read
file_name = "addition_floats.csv"
max_rows = 9000

# Read the CSV file
with open(file_name, mode='r', newline='') as file:
    reader = csv.reader(file)
    
    # Iterate over the rows in the file
    for i, row in enumerate(reader):
        # Break the loop after reading 9000 rows
        if i >= max_rows:
            break
        data.append([float(row[0]), float(row[1])])
        targets.append(float(row[2]))


# Training loop
errors = []
for epoch in range(25):
    for x1, x2, actual in zip(*zip(*data), targets):
        update_weights(x1, x2, actual)
        predicted = predict(x1, x2)
        errors.append(compute_error(actual, predicted))

    print(f"Epoch {epoch + 1}, Weights: {weights}")

# Plotting the error over iterations
plt.figure(figsize=(10, 6))
plt.plot(errors)
plt.title('Loss Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

print(predict(1, 2))

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