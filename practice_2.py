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